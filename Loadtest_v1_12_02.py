import random
import json
from shapely.geometry import *
import folium
from collections import namedtuple
from bisect import bisect
import numpy as np
from math import pi, sqrt, cos, inf, log10, floor
from vincenty import *
import time
import tracemalloc
from multiprocessing import Pool, cpu_count
import pandas as pd
import os
import linecache
from pymongo import MongoClient


colors_fill = ['red', 'yellow', 'orange', 'magenta', 'green', 'blue', 'purple', 'brown', 'gray', 'black']
fill_opacity = 0.3
def style_function(feature):

    lowfreq = feature['properties']['frequencyRange']['lowFrequency']
    cnt = int((lowfreq/1000000.0-3550.0)/15)
    if cnt < 0 or cnt > 9:
        cnt = 0

    return {
        'fillColor': colors_fill[cnt],
        'fillOpacity': fill_opacity,
        'color': colors_fill[cnt],
        'weight': 10,
    }

def style_functionblack(feature):
    return {
        'fillColor': 'black',
        'fillOpacity': fill_opacity,
        'color': 'black',
        'weight': 3
    }

def style_functionred(feature):
    return {
        'fillColor': 'red',
        'fillOpacity': fill_opacity,
        'color': 'red',
        'weight': 3
    }

def style_functionorange(feature):
    return {
        'fillColor': 'orange',
        'fillOpacity': fill_opacity,
        'color': 'orange',
        'weight': 3
    }

def style_functionblue(feature):
    return {
        'fillColor': 'blue',
        'fillOpacity': fill_opacity,
        'color': 'blue',
        'weight': 3
    }

def style_functiongreen(feature):
    return {
        'fillColor': 'green',
        'fillOpacity': fill_opacity,
        'color': 'green',
        'weight': 3
    }



def display_top(snapshot, key_type='lineno', limit=3):
    snapshot = snapshot.filter_traces((
        tracemalloc.Filter(False, "<frozen importlib._bootstrap>"),
        tracemalloc.Filter(False, "<unknown>"),
    ))
    top_stats = snapshot.statistics(key_type)

    print("Top %s lines" % limit)
    for index, stat in enumerate(top_stats[:limit], 1):
        frame = stat.traceback[0]
        # replace "/path/to/module/file.py" with "module/file.py"
        filename = os.sep.join(frame.filename.split(os.sep)[-2:])
        print("#%s: %s:%s: %.1f KiB"
              % (index, filename, frame.lineno, stat.size / 1024))
        line = linecache.getline(frame.filename, frame.lineno).strip()
        if line:
            print('    %s' % line)

    other = top_stats[limit:]
    if other:
        size = sum(stat.size for stat in other)
        print("%s other: %.1f KiB" % (len(other), size / 1024))
    total = sum(stat.size for stat in top_stats)
    print("Total allocated size: %.1f KiB" % (total / 1024))


def random_points_within(poly, num_points):
    min_x, min_y, max_x, max_y = poly.bounds
    points = []

    while len(points) < num_points:
        random_point = Point([random.uniform(min_x, max_x), random.uniform(min_y, max_y)])
        if random_point.within(poly):
            points.append(random_point)

    return points

def initializer():
    global powerconstant
    global azimuths
    global rads
    global THRESHOLD_PER_10MHZ

    THRESHOLD_PER_10MHZ = -96  # dBm/MHz
    LightSpeedC = 3e8
    Freq = 3625*1000000  #hz
    # PathLossExponent = 2.2  # select depending on scenario
    PathLossExponent = 3.00  # select depending on scenario
    Wavelength = LightSpeedC/Freq
    azimuths = np.arange(0.0, 360.0, 3.0)
    rads = azimuths * pi/180.0

    powerconstant = (10*PathLossExponent*log10(4*pi/Wavelength))

def GetStandardAntennaGains(hor_dirs, ant_azimuth=None, ant_beamwidth=None, ant_gain=0):
  """Computes the antenna gains from a standard antenna defined by beamwidth.
  See R2-SGN-20.
  This uses the standard 3GPP formula for pattern derivation from a given
  antenna 3dB cutoff beamwidth.
  Directions and azimuth are defined compared to the north in clockwise
  direction and shall be within [0..360] degrees.

  Inputs:
    hor_dirs:       Ray directions in horizontal plane (degrees).
                    Either a scalar or an iterable.
    ant_azimut:     Antenna azimuth (degrees).
    ant_beamwidth:  Antenna 3dB cutoff beamwidth (degrees).
                    If None, then antenna is isotropic (default).
    ant_gain:       Antenna gain (dBi).

  Returns:
    The CBSD antenna gains (in dB).
    Either a scalar if hor_dirs is scalar or an ndarray otherwise.
  """
  is_scalar = np.isscalar(hor_dirs)
  hor_dirs = np.atleast_1d(hor_dirs)

  if (ant_beamwidth is None or ant_azimuth is None or
      ant_beamwidth == 0 or ant_beamwidth == 360):
    gains = ant_gain * np.ones(hor_dirs.shape)
  else:
    bore_angle = hor_dirs - ant_azimuth
    bore_angle[bore_angle > 180] -= 360
    bore_angle[bore_angle < -180] += 360
    gains = -12 * (bore_angle / float(ant_beamwidth))**2
    gains[gains < -20] = -20.
    gains += ant_gain

  if is_scalar: return gains[0]
  return gains

def GenContour(AntennaParams):

    antenna_gains = GetStandardAntennaGains(azimuths, AntennaParams[0],
                    AntennaParams[1], AntennaParams[2])

    Pr0max = AntennaParams[3] - powerconstant
    d_g = []
    for cnt, g in enumerate(antenna_gains):
        d_g.append(10 ** ((Pr0max - AntennaParams[2] + g - THRESHOLD_PER_10MHZ) / 20.0))

    circle = []
    for cnt, az in enumerate(rads):  # consider simpify this
        lat, lon, bear = GeodesicPoint(AntennaParams[4], AntennaParams[5],
                                       d_g[cnt] / 1000, azimuths[cnt])
        circle.append([lat, lon])
    xx = [item[0] for item in circle]  # lats
    yy = [item[1] for item in circle]  # lons
    Coverage = Polygon(list(zip(yy, xx)))

    return Coverage


whatifCBSD_template = {'information': {'version': '1.0', 'owner': 'SASM', 'description': ''},
                       'sourceInfo': {'source': 'local', 'description': ''}, 'type': 'FeatureCollection', 'features': [
        {'type': 'Feature', 'geometry': {'type': 'Point', 'coordinates': [-118.281646, 33.738992]}, 'properties': {
            'admin': {'sasId': 'FEDERATEDWIRELESS1569559150139_6766309247914254134',
                      'cbsdId': '2AD8UFW2QADPM01EB1846130250',
                      'locationInfo': {'stateFP': '06', 'countyFP': '037', 'geoID': '06037296220', 'stateName': 'CA',
                                       'countyName': 'Los Angeles County'}, 'cbsdType': 'BaseStation',
                      'userId': 'CtlqPY', 'fccId': '2AD8UFW2QADPM01', 'cbsdSerialNumber': 'EB1846130250',
                      'callSign': 'WJ2XWN', 'cbsdCategory': 'A', 'airInterface': {'radioTechnology': 'E_UTRA'},
                      'installationParam': {'latitude': 33.738992, 'longitude': -118.281646, 'height': 1.0,
                                            'heightType': 'AGL', 'horizontalAccuracy': 1.0, 'verticalAccuracy': 1.0,
                                            'indoorDeployment': True, 'antennaAzimuth': 0, 'antennaDowntilt': -9999,
                                            'antennaGain': 0, 'eirpCapability': 24, 'antennaBeamwidth': 360,
                                            'antennaModel': 'OmniFW2QQD'},
                      'measCapability': ['RECEIVED_POWER_WITHOUT_GRANT'],
                      'cbsdInfo': {'vendor': 'Nokia', 'model': 'FR2QD',
                                   'softwareVersion': 'TLF18A_ENB_0000_020235_387955'}},
            'oper': {'haat': 1.7976931348623157e+308, 'blacklisted': False, 'receivedMeasReportInquiry': True,
                     'receivedMeasReportGrant': True, 'receivedMeasReportHeartbeat': True, 'state': 'REGISTERED',
                     'registrationTime': '2019-10-01T10:02:47Z', 'pathlossReady': True,
                     'ceRegistrationState': 'REGISTERED', 'tsVersion': 'v1.2',
                     'cbsdReferenceId': '2AD8UFW2QADPM01__f__d27872db9d2da96e1eb615dd580a97d67ceeafd6'},
            'channelMap': [{'eirp': 0.0, 'state': 'NEAR_INCUMBENT', 'palID': [], 'palId': ['PAL_ID0', 'PAL_ID11']},
                           {'eirp': 0.0, 'state': 'NEAR_INCUMBENT', 'palID': [], 'palId': ['PAL_ID2', 'PAL_ID13']},
                           {'eirp': 0.0, 'state': 'NEAR_INCUMBENT', 'palID': []},
                           {'eirp': 0.0, 'state': 'NEAR_INCUMBENT', 'palID': [], 'palId': ['PAL_ID4']},
                           {'eirp': 0.0, 'state': 'NEAR_INCUMBENT', 'palID': []},
                           {'eirp': 0.0, 'state': 'NEAR_INCUMBENT', 'palID': []},
                           {'eirp': 0.0, 'state': 'NEAR_INCUMBENT', 'palID': []},
                           {'eirp': 0.0, 'state': 'NEAR_INCUMBENT', 'palID': []},
                           {'eirp': 0.0, 'state': 'NEAR_INCUMBENT', 'palID': []},
                           {'eirp': 0.0, 'state': 'NEAR_INCUMBENT', 'palID': []},
                           {'eirp': 0.0, 'state': 'NEAR_INCUMBENT', 'palID': []},
                           {'eirp': 0.0, 'state': 'NEAR_INCUMBENT', 'palID': []},
                           {'eirp': 0.0, 'state': 'NEAR_INCUMBENT', 'palID': []},
                           {'eirp': 0.0, 'state': 'NEAR_INCUMBENT', 'palID': []},
                           {'eirp': 0.0, 'state': 'NEAR_INCUMBENT', 'palID': []},
                           {'eirp': 0.0, 'state': 'NEAR_INCUMBENT', 'palID': []},
                           {'eirp': 0.0, 'state': 'NEAR_INCUMBENT', 'palID': []},
                           {'eirp': 0.0, 'state': 'NEAR_INCUMBENT', 'palID': []},
                           {'eirp': 0.0, 'state': 'NEAR_INCUMBENT', 'palID': []},
                           {'eirp': 0.0, 'state': 'NEAR_INCUMBENT', 'palID': []},
                           {'eirp': 0.0, 'state': 'NEAR_INCUMBENT', 'palID': []},
                           {'eirp': 0.0, 'state': 'NEAR_INCUMBENT', 'palID': []},
                           {'eirp': 0.0, 'state': 'NEAR_INCUMBENT', 'palID': []},
                           {'eirp': 0.0, 'state': 'NEAR_INCUMBENT', 'palID': []},
                           {'eirp': 0.0, 'state': 'NEAR_INCUMBENT', 'palID': []},
                           {'eirp': 0.0, 'state': 'NEAR_INCUMBENT', 'palID': []},
                           {'eirp': 0.0, 'state': 'NEAR_INCUMBENT', 'palID': []},
                           {'eirp': 0.0, 'state': 'NEAR_INCUMBENT', 'palID': []},
                           {'eirp': 0.0, 'state': 'NEAR_INCUMBENT', 'palID': []},
                           {'eirp': 0.0, 'state': 'NEAR_INCUMBENT', 'palID': []}]}}]}

def createwhatifCBSDs(randomSeed, numGrantsTotal, fixedPointGenerator, userids, showmap, PAL_county_perc,
                      PAL_county_cbsd_perc, multiGrantsThreshold, BW_threshold, numHbs, hbInterval, fccId):
    # reset seed to allow repeatable results since seed after ids will vary depending on numcbsds

    random.seed(randomSeed)
    cbsdCnt = 0
    grantCnt = 0
    CBSDdata = []
    PAL_cbsd_num = 0
    # to allow repeatable results
    random.seed(randomSeed)
    ids = random.sample(range(100000, 999999), numGrantsTotal + 5)

    dictOfPAL = {}
    for i in userids:
        dictOfPAL[i] = []

    myclient = MongoClient()
    db = myclient['comm-prod-01-sasm']

    db['load_cbsd'].delete_many({})
    reg = db['load_cbsd']
    db['load_grants'].delete_many({})
    grants = db['load_grants']

    reg_setup = db['cbsd.registrations']
    reg_setup_list = list(reg_setup.find({}, {'_id': 0}))
    dummy_reg = []
    for r in reg_setup_list:
        dummy_reg.append(r)
        if len(dummy_reg) == 20:
            reg.insert_many(dummy_reg)
            dummy_reg = []
    if len(dummy_reg) > 0:
        reg.insert_many(dummy_reg)

    grant_setup = db['grant.sas']
    grant_setup_list = list(grant_setup.find({}, {'_id': 0}))
    dummy_grant = []

    for g in grant_setup_list:
        dummy_grant.append(g)
        if len(dummy_grant) == 20:
            grants.insert_many(dummy_grant)
            dummy_grant = []

    if len(dummy_grant) > 0:
        grants.insert_many(dummy_grant)

    whatifCBSD_template = list(reg.find({}).limit(1))
    print(whatifCBSD_template[0])
    whatifGRANT_template = list(grants.find({}).limit(1))
    print(whatifGRANT_template[0])

    '''
    Use the above template to assign the generated values for every CBSD and Grant and append them to 
    load test database

    '''

    if fixedPointGenerator == False:
        print("parsing OnLand.geojson")
        with open('OnLand.geojson', 'r') as f:
            geo_json_data_cnty = json.load(f)
        cntys = []
        population = []
        states = []
        counties = []

        # NEW
        whether_PAL = []

        for cnty_cnt, county in enumerate(geo_json_data_cnty['features']):
            onland = shape(county['geometry'])
            geoid_la = county["properties"]["GEOID"]
            cntys.append(onland)
            population.append(county['properties']["population"])
            states.append(county['properties']['STATEFP'])
            counties.append(county['properties']['COUNTYFP'])

            # NEW - DETERMINES WHETHER POSSIBLY PAL.  IF SO, NUMBERED 1-5
            # if random.random() <= 0.3:#30% PAL area
            if str(geoid_la) == "06037":
                print(county["properties"]["NAME"])
                whether_PAL.append(random.choice(userids))
            elif random.random() <= PAL_county_perc:
                whether_PAL.append(random.choice(userids))
            else:
                whether_PAL.append("GAA")

        cum_population = [sum(population[:x]) / sum(population) for x in range(1, 1 + len(population))]
        if showmap:
            hmap = folium.Map(location=[40, -90],
                              tiles="Stamen Terrain", zoom_start=8, control_scale=True)
    else:
        fixedPoints = namedtuple("fixedPoints", "x y")
        points = fixedPoints(x=43.6, y=-106.23)

    while grantCnt < numGrantsTotal:

        if cbsdCnt % 100 == 0:
            print("finished generating {} cbsds {} grants".format(cbsdCnt, grantCnt))

        if fixedPointGenerator == False:
            ran = random.random()
            pos = bisect(cum_population, ran)
            points = random_points_within(cntys[pos], 1)[0]

            st = states[pos]
            cnty = counties[pos]
            geoid = st + cnty
            # extract state

            if showmap:
                folium.CircleMarker([points.y, points.x], radius=5, color='blue').add_to(hmap)
        r = random.random()
        if r < 0.05:
            EIRP = 30
            height = 6
            BW = 360
            az = 0
            Indoor = True
            Cat = 'A'
        elif r < 0.25:
            EIRP = 35
            height = 6
            BW = 360
            az = 0
            Indoor = False
            Cat = 'B'
        elif r < 0.6:
            EIRP = 30
            height = 6
            BW = 360
            az = 0
            Indoor = False
            Cat = 'B'
        elif r < 0.75:
            EIRP = 35.5
            height = 20
            BW = 120
            az = random.randint(0, 359)
            Indoor = False
            Cat = 'B'
        elif r < 0.90:
            EIRP = 42.65
            height = 20
            BW = 120
            az = random.randint(0, 359)
            Indoor = False
            Cat = 'B'
        else:
            EIRP = 47
            height = 20
            BW = 120
            az = random.randint(0, 359)
            Indoor = False
            Cat = 'B'

        multiGrantFlag = random.uniform(0, 1) > multiGrantsThreshold

        numGrantsForCbsd = 1
        if multiGrantFlag == True:
            bandwidth = random.choices([10000000, 20000000], weights=[0.6, 0.4])[0]
            if bandwidth == 20000000:
                numGrantsForCbsd = random.choice([2, 3, 4])
            else:
                numGrantsForCbsd = random.choice([2, 3, 4, 5, 6, 7, 8, 9, 10])
            bandwidthOverall = bandwidth * numGrantsForCbsd
        else:
            bandwidth = random.choices([10000000, 20000000, 30000000, 40000000],
                                       weights=[2500 / 50500, 27000 / 50500, 13500 / 50500, 7500 / 50500])[0]
            bandwidthOverall = bandwidth

        if numGrantsForCbsd > 1:
            skipDeRegistration = True
        else:
            skipDeRegistration = False
        lower_freq_start = random.randrange(3550000000, 3700000000 - bandwidthOverall, 10000000)

        if BW == 360:
            serialNumber = 'inusacbsd' + str(ids[cbsdCnt])
            cbsdCnt += 1

            '''
            Add a CBSD info in here
            1. Make changes to the whatifCBSD_template
            reg.insert_one(whatifCBSD_template)
            '''

            skipRegistration = False
            for grantNum in range(0, numGrantsForCbsd):
                lower_freq = lower_freq_start + grantNum * bandwidth
                upper_freq = lower_freq + bandwidth
                waitTimeBeforeReg = 50 * grantNum
                if upper_freq > BW_threshold:
                    userId = "abcdef"
                tup = (points.x, points.y, Cat, height, az, -8,
                       6, BW, lower_freq, upper_freq, EIRP - 10,
                       Indoor, EIRP, True, EIRP - 10, numHbs, hbInterval, fccId, serialNumber, userId,
                       skipRegistration, skipDeRegistration, waitTimeBeforeReg, geoid)
                CBSDdata.append(tup)
                '''
                    Add a Grant info in here
                    1. Make changes to the whatifGRANT_template
                    grants.insert_one(whatifGRANT_template)
                '''
                grantCnt += 1
                skipRegistration = True
        else:
            # generate the 3 sectors
            serialNumber = 'inusacbsd' + str(ids[cbsdCnt])
            cbsdCnt += 1
            skipRegistration = False
            for grantNum in range(0, numGrantsForCbsd):
                lower_freq = lower_freq_start + grantNum * bandwidth
                upper_freq = lower_freq + bandwidth
                waitTimeBeforeReg = 50 * grantNum
                if upper_freq > BW_threshold:
                    userId = "abcdef"
                tup = (points.x, points.y, Cat, height, az, -8,
                       6, BW, lower_freq, upper_freq, EIRP - 10,
                       Indoor, EIRP, True, EIRP - 10, numHbs, hbInterval, fccId, serialNumber, userId,
                       skipRegistration, skipDeRegistration, waitTimeBeforeReg, geoid)
                CBSDdata.append(tup)
                grantCnt += 1
                skipRegistration = True
                if grantCnt >= numGrantsTotal:
                    break

            serialNumber = 'inusacbsd' + str(ids[cbsdCnt])
            cbsdCnt += 1
            skipRegistration = False
            for grantNum in range(0, numGrantsForCbsd):
                lower_freq = lower_freq_start + grantNum * bandwidth
                upper_freq = lower_freq + bandwidth
                waitTimeBeforeReg = 50 * grantNum
                if upper_freq > BW_threshold:
                    userId = "abcdef"
                tup = (points.x, points.y, Cat, height, (az + 120) % 360, -8,
                       6, BW, lower_freq, upper_freq, EIRP - 10,
                       Indoor, EIRP, True, EIRP - 10, numHbs, hbInterval, fccId, serialNumber, userId,
                       skipRegistration, skipDeRegistration, waitTimeBeforeReg, geoid)
                CBSDdata.append(tup)
                grantCnt += 1
                skipRegistration = True
                if grantCnt >= numGrantsTotal:
                    break

            serialNumber = 'inusacbsd' + str(ids[cbsdCnt])
            cbsdCnt += 1
            skipRegistration = False
            for grantNum in range(0, numGrantsForCbsd):
                lower_freq = lower_freq_start + grantNum * bandwidth
                upper_freq = lower_freq + bandwidth
                waitTimeBeforeReg = 50 * grantNum
                if upper_freq > BW_threshold:
                    userId = "abcdef"
                tup = (points.x, points.y, Cat, height, (az + 240) % 360, -8,
                       6, BW, lower_freq, upper_freq, EIRP - 10,
                       Indoor, EIRP, True, EIRP - 10, numHbs, hbInterval, fccId, serialNumber, userId,
                       skipRegistration, skipDeRegistration, waitTimeBeforeReg, geoid)
                CBSDdata.append(tup)
                grantCnt += 1

                skipRegistration = True
                if grantCnt >= numGrantsTotal:
                    break

        if grantCnt >= numGrantsTotal:
            break

    print("finished generating {} cbsds {} grants".format(cbsdCnt, grantCnt))

def Contour_main():
    with open('config.json', 'r') as j:
        config_file = json.load(j)

    numGrantsTotal = config_file["numGrants"]
    randomSeed = config_file["randomSeed"]
    numHbs = config_file["csvParams"]["numberOfHeartbeats"]
    hbInterval = config_file["csvParams"]["heartbeatInterval"]
    fccId = config_file["csvParams"]["fccId"]
    multiGrantsThreshold = config_file["multiGrantsThreshold"]
    fixedPointGenerator = config_file["fixedPointGenerator"]
    showmap = config_file["showOnMap"]
    PAL_county_perc = 0.3
    PAL_county_cbsd_perc = 0.3
    BW_threshold = 3620000000

    df = pd.read_csv("URIDtoFRNMapping.csv")
    URIDtoFRNMapping = {}
    userids = []
    for key, value in df.iterrows():
        URIDtoFRNMapping[str(key)] = list(value.values)

    for row in URIDtoFRNMapping.values():
        userid = row[1]
        if userid not in userids:
            userids.append(userid)


    createwhatifCBSDs(randomSeed, numGrantsTotal, fixedPointGenerator, userids, showmap, PAL_county_perc,
                      PAL_county_cbsd_perc, multiGrantsThreshold, BW_threshold, numHbs, hbInterval, fccId)




    myclient = MongoClient()
    db = myclient['comm-prod-01-sasm']
    grants = db['load_grants']
    whatifgrants = list(grants.find({}))
    print("The number of grants,",len(whatifgrants))
    ContourInfo = []

    start = time.time()
    for CBSD in whatifgrants:
        mycbsd = list(CBSD)
        if mycbsd[11]:  # Add 15 dB loss
            #ContourInfo.append([grant.antennaAzimuth, grant.antennaBeamwidth, grant.antennaGain, grant.iapEirp, grant.lat, grant.lon])
            ContourInfo.append([mycbsd[4], mycbsd[7], mycbsd[6], mycbsd[10]- 15, mycbsd[1],mycbsd[0]])
        else:
            ContourInfo.append([mycbsd[4], mycbsd[7], mycbsd[6], mycbsd[10], mycbsd[1],mycbsd[0]])

    pool = Pool(cpu_count(), initializer, ())
    contours = list(pool.map(GenContour, ContourInfo))
    pool.close()
    pool.join()

    #snapshot = tracemalloc.take_snapshot()
    #display_top(snapshot)

    end = time.time()
    print("Coverage Contours %s secs" % (end-start))
    CBSDcontours = []
    hmap2 = folium.Map(location=[40, -90], zoom_start=8, control_scale=True)
    for cnt, cbsd in enumerate(whatifgrants):
        #grant.contour = contours[cnt]
        c = folium.GeoJson(contours[cnt], name=cnt,
                           style_function=style_functionblue)
        c.add_to(hmap2)

    hmap2.save(outfile="mycbsd.html")
    print(list(whatifgrants[0]))

if __name__ == '__main__':
    Contour_main()