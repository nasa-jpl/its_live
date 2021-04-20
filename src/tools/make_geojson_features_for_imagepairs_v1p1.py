import numpy as np
import argparse
import geojson
import h5py
import pyproj
import s3fs
import os
import json
import sys
import psutil
import time

class memtracker:

    def __init__(self, include_time=True):
        self.output_time = include_time
        if include_time:
            self.start_time = time.time()
        self.process = psutil.Process()
        self.startrss = self.process.memory_info().rss
        self.startvms = self.process.memory_info().vms

    def meminfo(self, message):
        if self.output_time:
            time_elapsed_seconds = time.time() - self.start_time
            print(f'{message:<30}:  time: {time_elapsed_seconds:8.2f} seconds    mem_percent {self.process.memory_percent()} ' +
                    f'delrss={self.process.memory_info().rss - self.startrss:16,}    ' +
                    f'delvms={self.process.memory_info().vms - self.startvms:16,}',
                    flush=True)
        else: # don't output time
            print(f'{message:<30}:  delrss={self.process.memory_info().rss - self.startrss:16,}   mem_percent {self.process.memory_percent()} ' +
                    f'delvms={self.process.memory_info().vms - self.startvms:16,}',
                    flush=True)

mt = memtracker()

s3 = s3fs.S3FileSystem(anon=True)


# returns a string (N78W124) for directory name based on granule centerpoint lat,lon
#  !!!! Not implemented in geojson code yet !!! <- remove this line when it is.
def finddirstring(lat,lon):
    if lat >= 0.0:
        NShemi_str = 'N'
    else:
        NShemi_str = 'S'
    if lon >= 0.0:
        EWhemi_str = 'E'
    else:
        EWhemi_str = 'W'
    outlat = int(10*np.trunc(np.abs(lat/10.0)))
    if outlat == 90: # if you are exactly at a pole, put in lat = 80 bin
        outlat = 80
    outlon = int(10*np.trunc(np.abs(lon/10.0)))
    if outlon >= 180: # if you are at the dateline, back off to the 170 bin
        outlon = 170
    dirstring = f'{NShemi_str}{outlat:02d}{EWhemi_str}{outlon:03d}'
    return(dirstring)

def image_pair_feature_from_path(infilewithpath, five_points_per_side = False):
    # from s3.ls:
    #     infilewithpath = 'https://s3/its-live-data.jpl.nasa.gov/velocity_image_pair/landsat/v00.0/32609/LC08_L1TP_050024_20180713_20180730_01_T1_X_LE07_L1TP_050024_20180315_20180316_01_RT_G0240V01_P072.nc'


    # base URL from S3 directory listing has file path for s3fs access, not what you need for http directly,
    #  so that is hard coded here. (or not used - don't need it in every feature)
    # base_URL = 'http://its-live-data.jpl.nasa.gov.s3.amazonaws.com/velocity_image_pair/landsat/v00.0'

    directory,filename = infilewithpath.split('/')[-2:]

    # infilewithpath = 'LC08_L1TP_050024_20180814_20180828_01_T1_X_LC08_L1TP_050024_20170928_20171013_01_T1_G0240V01_P091.nc'
    #     inh5 = h5py.File(infilewithpath, mode = 'r')

    with s3.open(f"s3://{infilewithpath}", "rb") as ins3:
        inh5 = h5py.File(ins3, mode = 'r')
        # inh5 = h5py.File(s3.open(f"s3://{infilewithpath}", "rb"), mode = 'r')
        #     inh5 = h5py.File(ingeoimg.in_dir_path + '/' + ingeoimg.filename,mode='r')
        # netCDF4/HDF5 cf 1.6 has x and y vectors of array pixel CENTERS
        xvals = np.array(inh5.get('x'))
        yvals = np.array(inh5.get('y'))

        # Extract projection variable
        projection_cf = inh5['UTM_Projection'] if 'UTM_Projection' in inh5 else inh5['Polar_Stereographic']

        imginfo_attrs = inh5['img_pair_info'].attrs
        # turn hdf5 img_pair_info attrs into a python dict to save below
        img_pair_info_dict = {}
        for k in imginfo_attrs.keys():
            if imginfo_attrs[k].shape == ():
                img_pair_info_dict[k] = imginfo_attrs[k].decode('utf-8')  # h5py returns byte values, turn into byte characters
            else:
                img_pair_info_dict[k] = imginfo_attrs[k][0]    # h5py returns lists of numbers - all 1 element lists here, so dereference to number

        num_pix_x = len(xvals)
        num_pix_y = len(yvals)

        minval_x, pix_size_x, rot_x_ignored, maxval_y, rot_y_ignored, pix_size_y = [float(x) for x in projection_cf.attrs['GeoTransform'].split()]

        epsgcode = int(projection_cf.attrs['spatial_epsg'][0])
        inh5.close()

    # NOTE: these are pixel center values, need to modify by half the grid size to get bounding box/geotransform values
    projection_cf_minx = xvals[0] - pix_size_x/2.0
    projection_cf_maxx = xvals[-1] + pix_size_x/2.0
    projection_cf_miny = yvals[-1] + pix_size_y/2.0 # pix_size_y is negative!
    projection_cf_maxy = yvals[0] - pix_size_y/2.0  # pix_size_y is negative!


    transformer = pyproj.Transformer.from_crs(f"EPSG:{epsgcode}", "EPSG:4326", always_xy=True) # ensure lonlat output order

    ll_lonlat = np.round(transformer.transform(projection_cf_minx,projection_cf_miny),decimals = 7).tolist()
    lr_lonlat = np.round(transformer.transform(projection_cf_maxx,projection_cf_miny),decimals = 7).tolist()
    ur_lonlat = np.round(transformer.transform(projection_cf_maxx,projection_cf_maxy),decimals = 7).tolist()
    ul_lonlat = np.round(transformer.transform(projection_cf_minx,projection_cf_maxy),decimals = 7).tolist()

    # find center lon lat for inclusion in feature (to determine lon lat grid cell directory)
#     projection_cf_centerx = (xvals[0] + xvals[-1])/2.0
#     projection_cf_centery = (yvals[0] + yvals[-1])/2.0
    center_lonlat = np.round(transformer.transform((xvals[0] + xvals[-1])/2.0,(yvals[0] + yvals[-1])/2.0 ),decimals = 7).tolist()

    if five_points_per_side:
        fracs = [0.25, 0.5, 0.75]
        polylist = [] # ring in counterclockwise order

        polylist.append(ll_lonlat)
        dx = projection_cf_maxx - projection_cf_minx
        dy = projection_cf_miny - projection_cf_miny
        for frac in fracs:
            polylist.append(np.round(transformer.transform(projection_cf_minx + (frac * dx), projection_cf_miny + (frac * dy)),decimals = 7).tolist())

        polylist.append(lr_lonlat)
        dx = projection_cf_maxx - projection_cf_maxx
        dy = projection_cf_maxy - projection_cf_miny
        for frac in fracs:
            polylist.append(np.round(transformer.transform(projection_cf_maxx + (frac * dx), projection_cf_miny + (frac * dy)),decimals = 7).tolist())

        polylist.append(ur_lonlat)
        dx = projection_cf_minx - projection_cf_maxx
        dy = projection_cf_maxy - projection_cf_maxy
        for frac in fracs:
            polylist.append(np.round(transformer.transform(projection_cf_maxx + (frac * dx), projection_cf_maxy + (frac * dy)),decimals = 7).tolist())

        polylist.append(ul_lonlat)
        dx = projection_cf_minx - projection_cf_minx
        dy = projection_cf_miny - projection_cf_maxy
        for frac in fracs:
            polylist.append(np.round(transformer.transform(projection_cf_minx + (frac * dx), projection_cf_maxy + (frac * dy)),decimals = 7).tolist())

        polylist.append(ll_lonlat)

    else:
        # only the corner points
        polylist = [ ll_lonlat, lr_lonlat, ur_lonlat, ul_lonlat, ll_lonlat ]

    poly = geojson.Polygon([polylist])

    middate = img_pair_info_dict['date_center']
    deldays = img_pair_info_dict['date_dt']
    percent_valid_pix = img_pair_info_dict['roi_valid_percentage']

    feat = geojson.Feature( geometry=poly,
                            properties={
                                        'filename': filename,
                                        'directory': directory,
                                        'middate':middate,
                                        'deldays':deldays,
                                        'percent_valid_pix': percent_valid_pix,
                                        'center_lonlat':center_lonlat,
                                        'data_epsg':epsgcode,
                                        # date_deldays_strrep is a string version of center date and time interval that will sort by date and then by interval length (shorter intervals first) - relies on "string" comparisons by byte
                                        'date_deldays_strrep': img_pair_info_dict['date_center'] + f"{img_pair_info_dict['date_dt']:07.1f}".replace('.',''),
                                        'img_pair_info_dict': img_pair_info_dict,
                                        }
                            )
    return(feat)





parser = argparse.ArgumentParser( \
    description="""make_geojson_features_for_imagepairs_v1.py

                    produces output geojson FeatureCollection for each nn image_pairs from a zone.
                    v1 adds 5 points per side to geom (so 3 interior and the two corners from v0)
                    and the ability to stop the chunks (in addition to the start allowed in v0)
                    so that the code can be run on a range of chunks.
                    """,
    epilog='',
    formatter_class=argparse.RawDescriptionHelpFormatter)

parser.add_argument('-base_dir_s3fs',
                    action='store',
                    type=str,
                    default='its-live-data.jpl.nasa.gov/velocity_image_pair/landsat/v00.0',
                    help='S3 path to tile catalog directories (not including the EPSG code for zone of tile) [%(default)s]')

parser.add_argument('-S3_output_directory',
                    action='store',
                    type=str,
                    default='its-live-data.jpl.nasa.gov/test_geojson_catalog',
                    help='output path for featurecollections [%(default)s]')

parser.add_argument('-read_filelist_from_S3_file',
                    action='store',
                    type=str,
                    default=None,
                    help='get input file list from S3 file here: [None]')

parser.add_argument('-base_download_URL',
                    action='store',
                    type=str,
                    default='http://its-live-data.jpl.nasa.gov.s3.amazonaws.com/velocity_image_pair/landsat/v00.0',
                    help='URL base for download of image pair - need zone(directory) and filename as well [%(default)s]')
parser.add_argument('-chunk_by',
                    action='store',
                    type=int,
                    default=20000,
                    help='chunk feature collections to have chunk_by features each [%(default)d]')
parser.add_argument('-start_chunks_at_file',
                    action='store',
                    type=int,
                    default=0,
                    help='start run at chunk that begins at file n [%(default)d]')
parser.add_argument('-stop_chunks_at_file',
                    action='store',
                    type=int,
                    default=0,
                    help='stop run just befor chunk that begins at file n [%(default)d]')
args = parser.parse_args()


inzonesdir = args.base_dir_s3fs

if args.read_filelist_from_S3_file:
    # read in infiles from S3 file
    with s3.open(args.read_filelist_from_S3_file,'r') as ins3file:
        infilelist = json.load(ins3file)
else:
    # use a glob to list directory
    infilelist = s3.glob(f'{inzonesdir}/*.nc')

# check for '_P' in filename - filters out temp.nc files that can be left by bad transfers
# also skips txt file placeholders for 000 Pct (all invalid) pairs
infiles = [x for x in infilelist if '_P' in x and 'txt' not in x]

totalnumfiles = len(infiles)

mt.meminfo(f'working on {totalnumfiles} total files from {inzonesdir}')


# set up tuples of start,stop indicies in file list for chunk processing
numout_featuresets = np.round(totalnumfiles/args.chunk_by).astype('int')
if numout_featuresets == 0:
    if totalnumfiles == 0:
        print(f'No files found for {inzonesdir}, exiting...')
        sys.exit(0)
    else:
        chunks_startstop = [(0, totalnumfiles-1)]
else:
    if numout_featuresets==1:
        chunks_startstop = [(0, totalnumfiles-1)]
    else:
        chunks_startstop = [((i)*args.chunk_by,((i+1) * args.chunk_by)-1) for i in range(numout_featuresets - 1)]
        chunks_startstop.append(((numout_featuresets - 1) * args.chunk_by, totalnumfiles-1))

# find start, and if specified, stop chunks in this list of tuples
if args.start_chunks_at_file != 0:
    new_chunks_startstop = [(x,y) for x,y in chunks_startstop if x >= args.start_chunks_at_file]
    if new_chunks_startstop[0][0] == args.start_chunks_at_file:
        chunks_startstop = new_chunks_startstop
    else:
        print(f'-start_chunks_at_file {args.start_chunks_at_file} not in {chunks_startstop}, quitting...')
        sys.exit(0)

if args.stop_chunks_at_file != 0:
    new_chunks_startstop = [(x,y) for x,y in chunks_startstop if x < args.stop_chunks_at_file]
    if new_chunks_startstop[-1][0] + args.chunk_by == args.stop_chunks_at_file:
        chunks_startstop = new_chunks_startstop
    else:
        print(f'-stop_chunks_at_file {args.stop_chunks_at_file} not in {chunks_startstop}, quitting...')
        sys.exit(0)

# Use sub-directory name of input path as base for output filename
base_dir = os.path.basename(inzonesdir)
s3_out = s3fs.S3FileSystem()

for num,(start,stop) in enumerate(chunks_startstop):
    print(f'working on chunk {start},{stop}', flush = True)
    featurelist = []
    count = start
    for infilewithpath in infiles[start:stop+1]:
        count += 1
        if count%100 == 0:
            if count%1000 == 0:
                mt.meminfo(f'{count:6d}/{stop:6d}')
            else:
                print(f'{count:6d}/{stop:6d}', end = '\r', flush = True)
        feature = image_pair_feature_from_path(infilewithpath, five_points_per_side = True)
        featurelist.append(feature)

    featureColl = geojson.FeatureCollection(featurelist)
    outfilename = f'imgpr_{base_dir}_{start:06d}_{stop:06d}.json'

    with s3_out.open(f'{args.S3_output_directory}/{outfilename}','w') as outf:
        geojson.dump(featureColl,outf)

    mt.meminfo(f'wrote {args.S3_output_directory}/{outfilename}')
    featurelist = None
    featureColl = None
