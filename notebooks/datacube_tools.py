# to get and use geojson datacube catalog
import s3fs as s3
import json
from shapely import geometry

# for timing data access
import time
import traceback


import numpy as np
import pyproj
# import pandas as pd

# for datacube xarray/zarr access
import xarray as xr

# for plotting time series
from matplotlib import pyplot as plt



# class to throw time series lookup errors
class timeseriesException(Exception):
    print(traceback.format_exc())
    pass





#         self.catalog = {
#             "all": "s3://its-live-data/datacubes/catalog_v02.json"
#         }

class DATACUBES:
    """
    class to encapsulate discovery and interaction with ITS_LIVE (its-live.jpl.nasa.gov) datacubes on AWS s3
    """

    VELOCITY_DATA_ATTRIBUTION = """ \nITS_LIVE velocity data
    (<a href="https://its-live.jpl.nasa.gov">ITS_LIVE</a>) with funding provided by NASA MEaSUREs.\n
    """

    def __init__(self, use_catalog='all'):
        """
        tools for accessing ITS_LIVE glacier velocity datacubes in S3
        """
        
        # the URL for the current datacube catalog GeoJSON file - set up as dictionary to allow other catalogs for testing
        self.catalog = {
            "all": "s3://its-live-data/datacubes/catalog_v02.json"
        }

        # S3fs used to access cubes in python
        self._s3fs = s3.S3FileSystem(anon=True)
        # keep track of open cubes so that we don't re-read xarray metadata and dimension vectors
        self.open_cubes = {}
        
        self._current_catalog = use_catalog
        with self._s3fs.open(self.catalog[use_catalog], "r") as incubejson:
            self._json_all = json.load(incubejson)
        self.json_catalog = self._json_all


    def find_datacube_catalog_entry_for_point(self, point_xy, point_epsg_str):
        """
        find catalog feature that contains the point_xy [x,y] in projection point_epsg_str
        returns the catalog feature and the point_tilexy original point coordinates reprojected into the datacube's native projection 
        (cubefeature, point_tilexy)
        """
        if point_epsg_str != "4326":
            # point not in lon,lat, set up transformation and convert it to lon,lat (epsg:4326)
            # because the features in the catalog GeoJSON are polygons in 4326
            inPROJtoLL = pyproj.Transformer.from_proj(
                f"epsg:{point_epsg_str}", "epsg:4326", always_xy=True
            )
            pointll = inPROJtoLL.transform(*point_xy)
        else:
            # point already lon,lat
            pointll = point_xy

        # create Shapely point object for inclusion test
        point = geometry.Point(*pointll)  # point.coords.xy

        # find datacube outline that contains this point in geojson index file
        cubefeature = None

        for f in self.json_catalog["features"]:
            polygeom = geometry.shape(f["geometry"])
            if polygeom.contains(point):
                cubefeature = f
                break

        if cubefeature:
            
            # find point x and y in cube native epsg if not already in that projection
            if point_epsg_str == cubefeature["properties"]["data_epsg"].split(':')[-1]:
                point_cubexy = point_xy
            else:
                inPROJtoTilePROJ = pyproj.Transformer.from_proj(
                    f"epsg:{point_epsg_str}",
                    cubefeature["properties"]["data_epsg"],
                    always_xy=True,
                )
                point_cubexy = inPROJtoTilePROJ.transform(*point_xy)

            print(
                f"original xy {point_xy} {point_epsg_str} maps to datacube {point_cubexy} "
                f" {cubefeature['properties']['data_epsg']}"
            )

            # now test if point is in xy box for cube (should be most of the time; could fail
            # because of boundary curvature 4326 box defined by lon,lat corners but point needs to be in box defined in cube's projection)
            #
            point_cubexy_shapely = geometry.Point(*point_cubexy)
            polygeomxy = geometry.shape(cubefeature["properties"]["geometry_epsg"])
            if not polygeomxy.contains(point_cubexy_shapely):
                # point is in lat lon box, but not in cube-projection's box
                # try once more to find proper cube by using a new point in cube projection moved 10 km farther from closest
                # boundary in cube projection; use new point's lat lon to search for new cube - test if old point is in that
                # new cube's projection box, otherwise ...
                
                # this next section tries one more time to find new feature after offsetting point farther outside box of 
                # first cube, in cube projection, to deal with curvature of lat lon box edges in different projections
                #
                # point in ll box but not cube_projection box, move point in cube projection
                # 10 km farther outside box, find new ll value for point, find new feature it is in,
                # and check again if original point falls in this new cube's 
                
                # first find cube proj bounding box, then move coordinate of point outside this box farther out by 10 km
                dcbbox = np.array(cubefeature["properties"]["geometry_epsg"]["coordinates"][0])
                minx = np.min(dcbbox[:, 0])
                maxx = np.max(dcbbox[:, 0])
                miny = np.min(dcbbox[:, 1])
                maxy = np.max(dcbbox[:, 1])


                newpoint_cubexy = point_cubexy
                if point_cubexy[1] < miny:
                    newpoint_cubexy[1] -= 10000.0
                elif point_cubexy[1] > maxy:
                    newpoint_cubexy[1] += 10000.0
                elif point_cubexy[0] < minx:
                    newpoint_cubexy[0] -= 10000.0
                elif point_cubexy[0] > maxx:
                    newpoint_cubexy[0] += 10000.0
                else:
                    # should never get here based on point not in box test at start
                    pass
                
                # now reproject this point to lat lon and look for new feature

                cubePROJtoLL = pyproj.Transformer.from_proj(
                    f'{cubefeature["properties"]["data_epsg"]}', "epsg:4326", always_xy=True
                )
                newpointll = cubePROJtoLL.transform(*newpoint_cubexy)

                # create Shapely point object for inclusion test
                newpoint = geometry.Point(*newpointll) 

                # find datacube outline that contains this point in geojson index file
                newcubefeature = None

                for f in self.json_catalog["features"]:
                    polygeom = geometry.shape(f["geometry"])
                    if polygeom.contains(newpoint):
                        newcubefeature = f
                        break

                if newcubefeature:
                    # if new feature found, see if original (not offset) point is in this new cube's cube-projection bounding box
                    # find point x and y in cube native epsg if not already in that projection
                    if cubefeature["properties"]["data_epsg"] == newcubefeature["properties"]["data_epsg"]:
                        pass   # point_cubexy stays the same for the original point
                    else:
                        # project original point in this new cube's projection
                        inPROJtoTilePROJ = pyproj.Transformer.from_proj(
                            f"epsg:{point_epsg_str}",
                            newcubefeature["properties"]["data_epsg"],
                            always_xy=True,
                        )
                        point_cubexy = inPROJtoTilePROJ.transform(*point_xy)

                    print(
                        f"try 2 original xy {point_xy} {point_epsg_str} maps to new datacube {point_cubexy} "
                        f" {newcubefeature['properties']['data_epsg']}"
                    )

                    # now test if point is in xy box for cube (should be most of the time; 
                    #
                    point_cubexy_shapely = geometry.Point(*point_cubexy)
                    polygeomxy = geometry.shape(newcubefeature["properties"]["geometry_epsg"])
                    if not polygeomxy.contains(point_cubexy_shapely):
                        # point is in lat lon box, but not in cube-projection's box
                        # try once more to find proper cube by using a new point in cube projection moved 10 km farther from closest
                        # boundary in cube projection; use new point's lat lon to search for new cube - test if old point is in that
                        # new cube's projection box, otherwise fail...
        
                        raise timeseriesException(
                            f"point is in lat,lon box but not {cubefeature['properties']['data_epsg']} box!! even after offset"
                        )
                    else:
                        return(newcubefeature, point_cubexy)

            else:
                return(cubefeature, point_cubexy)

        else:
            # raise timeseriesException(f"no datacube found for point {pointll}")
            print(f"No data for point (lon,lat) {pointll}")
            return (None, None)
               
                

    def get_timeseries(self, point_xy, point_epsg_str, variable):

        start = time.time()
            
        cube_feature,point_cubexy = self.find_datacube_catalog_entry_for_point(point_xy, point_epsg_str)
        
        # for zarr store modify URL for use in boto open - change http: to s3: and lose s3.amazonaws.com
        cube_feature = (
            cubef["properties"]["zarr_url"]
            .replace("http:", "s3:")
            .replace(".s3.amazonaws.com", "")
        )

        # if we have already opened this cube, don't do it again
        if len(self.open_cubes) > 0 and incubeurl in self.open_cubes.keys():
            ins3xr = self.open_cubes[incubeurl]
        else:
            ins3xr = xr.open_dataset(
                incubeurl, engine="zarr", storage_options={"anon": True}
            )
            self.open_cubes[incubeurl] = ins3xr

        pt_variable = ins3xr[variable].sel(
            x=point_cubexy[0], y=point_cubexy[1], method="nearest"
        )

        print(
            f"xarray open - elapsed time: {(time.time()-start):10.2f}", flush=True
        )

        pt_variable.load()

        print(
            f"time series loaded {len(pt_variable)} points - elapsed time: {(time.time()-start):10.2f}",
            flush=True,
        )
        # end for zarr store

        return (ins3xr, pt_variable, point_cubexy)
            
            
            
            
            
            
            
            
            
            

# class ITSLIVE:
#     """
#     Class to encapsulate ITS_LIVE plotting from zarr in S3
#     """
# 
#     VELOCITY_ATTRIBUTION = """ \nITS_LIVE velocity mosaic
#     (<a href="https://its-live.jpl.nasa.gov">ITS_LIVE</a>) with funding provided by NASA MEaSUREs.\n
#     """
# 
#     def __init__(self, *args, **kwargs):
#         """
#         Map widget to plot glacier velocities
#         """
#         self.catalog = {
#             "all": "s3://its-live-data/datacubes/catalog_v02.json"
#         }
#         self.config = {"plot": "v", "max_separation_days": 90, "color_by": "points"}
#         self._s3fs = s3.S3FileSystem(anon=True)
#         self.open_cubes = {}
#         # self.outwidget = ipywidgets.Output(layout={"border": "1px solid blue"})
# 
#         self.color_index = 0
#         self.icon_color_index = 0
#         self._last_click = None
# 
#         self._current_catalog = "All Satellites"
#         with self._s3fs.open(self.catalog["all"], "r") as incubejson:
#             self._json_all = json.load(incubejson)
#         self.json_catalog = self._json_all
#         self._initialize_widgets()
# 
#     def set_config(self, config):
#         self.config = config
# 
#     def _initialize_widgets(self):
#         self._control_plot_running_mean_checkbox = ipywidgets.Checkbox(
#             value=True,
#             description="Include running mean",
#             disabled=False,
#             indent=False,
#             tooltip="Plot running mean through each time series",
#             layout=ipywidgets.Layout(width="150px"),
#         )
#         self._control_plot_running_mean_widgcntrl = ipyleaflet.WidgetControl(
#             widget=self._control_plot_running_mean_checkbox, position="bottomright"
#         )
#         self._control_clear_points_button = ipywidgets.Button(
#             description="Clear Points", tooltip="clear all picked points"
#         )
#         self._control_clear_points_button.on_click(self.clear_points)
# 
#         self._control_clear_points_button_widgcntrl = ipyleaflet.WidgetControl(
#             widget=self._control_clear_points_button, position="bottomright"
#         )
# 
#         self._control_plot_button = ipywidgets.Button(
#             description="Make Plot", tooltip="click to make plot"
#         )
#         self._control_plot_button.on_click(self.plot_time_series)
#         self._control_plot_button_widgcntrl = ipyleaflet.WidgetControl(
#             widget=self._control_plot_button, position="bottomright"
#         )
# 
#         image = Image(
#             (
#                 "https://its-live-data.s3.amazonaws.com/documentation/"
#                 "ITS_LIVE_logo_small.png"
#             ),
#             width=220,
#         )
# 
#         self._control_logo = ipywidgets.Image(
#             value=image.data, format="png", width=180, height=58
#         )
#         self._control_logo_widgcntrl = ipyleaflet.WidgetControl(
#             widget=self._control_logo, position="topright"
#         )
#         self._map_base_layer = ipyleaflet.basemap_to_tiles(
#             {
#                 "url": (
#                     "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/"
#                     "MapServer/tile/{z}/{y}/{x}.jpg"
#                 ),
#                 "attribution": "\nImagery provided by ESRI\n",
#                 "name": "ESRI basemap",
#             }
#         )
#         self._map_velocity_layer = ipyleaflet.basemap_to_tiles(
#             {
#                 "url": "https://glacierflow.nyc3.digitaloceanspaces.com/webmaps/vel_map/{z}/{x}/{y}.png",
#                 "attribution": self.VELOCITY_ATTRIBUTION,
#                 "name": "ITS_LIVE Velocity Mosaic",
#             }
#         )
#         self._map_coverage_layer = ipyleaflet.GeoJSON(
#             data=self.json_catalog,
#             name="ITS_LIVE datacube coverage",
#             style={
#                 "opacity": 0.8,
#                 "fillOpacity": 0.2,
#                 "weight": 1,
#                 "color": "red",
#                 "cursor": "crosshair",
#             },
#             hover_style={
#                 "color": "white",
#                 "dashArray": "0",
#                 "fillOpacity": 0.5,
#             },
#         )
#         self.map = ipyleaflet.Map(
#             basemap=self._map_base_layer,
#             double_click_zoom=False,
#             scroll_wheel_zoom=True,
#             center=[64.20, -49.43],
#             zoom=3,
#             # layout=ipywidgets.widgets.Layout(
#             #     width="100%",  # Set Width of the map, examples: "100%", "5em", "300px"
#             #     height="100%",  # Set height of the map
#             # ),
#         )
#         self._map_picked_points_layer_group = ipyleaflet.LayerGroup(
#             layers=[], name="Picked points"
#         )
# 
#         # Populating the map
# 
#         self.map.add_layer(self._map_picked_points_layer_group)
#         self.map.add_layer(self._map_velocity_layer)
#         self.map.add_layer(self._map_coverage_layer)
#         self.map.add_control(
#             ipyleaflet.MeasureControl(
#                 position="topleft",
#                 active_color="orange",
#                 primary_length_unit="kilometers",
#             )
#         )
#         self.map.add_control(ipyleaflet.FullScreenControl())
#         self.map.add_control(ipyleaflet.LayersControl())
#         self.map.add_control(ipyleaflet.ScaleControl(position="bottomleft"))
#         self.map.add_control(self._control_plot_running_mean_widgcntrl)
#         self.map.add_control(self._control_clear_points_button_widgcntrl)
#         self.map.add_control(self._control_plot_button_widgcntrl)
#         self.map.add_control(self._control_logo_widgcntrl)
#         self.map.default_style = {"cursor": "crosshair"}
#         self.map.on_interaction(self._handle_map_click)
# 
#     def display(self, render_sidecar=True):
# 
#         if not hasattr(self, "sidecar"):
#             self.sidecar = Sidecar(title="Map Widget")
# 
#         if render_sidecar:
#             self.fig, self.ax = plt.subplots(1, 1, figsize=(10, 6))
#             self.sidecar.clear_output()
#             with self.sidecar:
#                 display(self.map)
# 
# 
#     def get_timeseries(self, point_xy, point_epsg_str, variable):
# 
#         start = time.time()
# 
#         if point_epsg_str != "4326":
#             # point not in lon,lat, set up transformation and convert it to lon,lat (epsg:4326)
#             inPROJtoLL = pyproj.Transformer.from_proj(
#                 f"epsg:{point_epsg_str}", "epsg:4326", always_xy=True
#             )
#             pointll = inPROJtoLL.transform(*point_xy)
#         else:
#             # point already lon,lat
#             pointll = point_xy
# 
#         # create Shapely point object for inclusion test
#         point = geometry.Point(*pointll)  # point.coords.xy
# 
#         # find datacube outline that contains this point in geojson index file
#         cubef = None
# 
#         # TODO: this should be done via the API
#         for f in self.json_catalog["features"]:
#             polygeom = geometry.shape(f["geometry"])
#             if polygeom.contains(point):
#                 cubef = f
#                 break
# 
#         if cubef:
#             print(
#                 f"found datacube - elapsed time: {(time.time()-start):10.2f}",
#                 flush=True,
#             )
# 
#             if point_epsg_str == cubef["properties"]["data_epsg"]:
#                 point_tilexy = point_xy
#             else:
#                 inPROJtoTilePROJ = pyproj.Transformer.from_proj(
#                     f"epsg:{point_epsg_str}",
#                     cubef["properties"]["data_epsg"],
#                     always_xy=True,
#                 )
#                 point_tilexy = inPROJtoTilePROJ.transform(*point_xy)
# 
#             print(
#                 f"original xy {point_xy} {point_epsg_str} maps to datacube {point_tilexy} "
#                 f" {cubef['properties']['data_epsg']}"
#             )
# 
#             # now test if point is in xy box for cube (should be most of the time; could fail
#             # because of boundary curvature 4326 box defined by lon,lat corners but point chosen in basemap projection)
#             point_tilexy_shapely = geometry.Point(*point_tilexy)
#             polygeomxy = geometry.shape(cubef["properties"]["geometry_epsg"])
#             if not polygeomxy.contains(point_tilexy_shapely):
#                 raise timeseriesException(
#                     f"point is in lat,lon box but not {cubef['properties']['data_epsg']} box!!"
#                 )
# 
#             # for zarr store modify URL for use in boto open - change http: to s3: and lose s3.amazonaws.com
#             incubeurl = (
#                 cubef["properties"]["zarr_url"]
#                 .replace("http:", "s3:")
#                 .replace(".s3.amazonaws.com", "")
#             )
# 
#             # if we have already opened this cube, don't do it again
#             if len(self.open_cubes) > 0 and incubeurl in self.open_cubes.keys():
#                 ins3xr = self.open_cubes[incubeurl]
#             else:
#                 ins3xr = xr.open_dataset(
#                     incubeurl, engine="zarr", storage_options={"anon": True}
#                 )
#                 self.open_cubes[incubeurl] = ins3xr
# 
#             pt_variable = ins3xr[variable].sel(
#                 x=point_tilexy[0], y=point_tilexy[1], method="nearest"
#             )
# 
#             print(
#                 f"xarray open - elapsed time: {(time.time()-start):10.2f}", flush=True
#             )
# 
#             pt_variable.load()
# 
#             print(
#                 f"time series loaded {len(pt_variable)} points - elapsed time: {(time.time()-start):10.2f}",
#                 flush=True,
#             )
#             # end for zarr store
# 
#             return (ins3xr, pt_variable, point_tilexy)
# 
#         else:
#             # raise timeseriesException(f"no datacube found for point {pointll}")
#             print(f"No data for point {pointll}")
#             return (None, None, None)
# 
#     # running mean
#     def runningMean(
#         self,
#         mid_dates,
#         variable,
#         minpts,
#         tFreq,
#     ):
#         """
#         mid_dates: center dates of `variable` data [datetime64]
#         variable: data to be average
#         minpts: minimum number of points needed for a valid value, else filled with nan
#         tFreq: the spacing between centered averages in Days, default window size = tFreq*2
#         """
#         tsmin = pd.Timestamp(np.min(mid_dates))
#         tsmax = pd.Timestamp(np.max(mid_dates))
#         ts = pd.date_range(start=tsmin, end=tsmax, freq=f"{tFreq}D")
#         ts = pd.to_datetime(ts).values
#         idx0 = ~np.isnan(variable)
#         runmean = np.empty([len(ts) - 1, 1])
#         runmean[:] = np.nan
#         tsmean = ts[0:-1]
# 
#         t_np = mid_dates.astype(np.int64)
# 
#         for i in range(len(ts) - 1):
#             idx = (
#                 (mid_dates >= (ts[i] - np.timedelta64(int(tFreq / 2), "D")))
#                 & (mid_dates < (ts[i + 1] + np.timedelta64(int(tFreq / 2), "D")))
#                 & idx0
#             )
#             if sum(idx) >= minpts:
#                 runmean[i] = np.mean(variable[idx])
#                 tsmean[i] = np.mean(t_np[idx])
# 
#         tsmean = pd.to_datetime(tsmean).values
#         return (runmean, tsmean)
# 
#     def _handle_map_click(self, **kwargs):
#         if kwargs.get("type") == "click":
#             # NOTE this is the work around for the double click issue discussed above!
#             # Only acknoledge the click when it is registered the second time at the same place!
#             if self._last_click and (
#                 kwargs.get("coordinates") == self._last_click.get("coordinates")
#             ):
#                 color = plt.cm.tab10(self.icon_color_index)
#                 print(self.icon_color_index, color)
#                 html_for_marker = f"""
#                 <div style="width: 3rem;height: 3rem; display: block;position: relative;transform: rotate(45deg);"/>
#                   <h1 style="position: relative;left: -2.5rem;top: -2.5rem;font-size: 3rem;">
#                     <span style="color: rgba({color[0]*100}%,{color[1]*100}%,{color[2]*100}%, {color[3]})">
#                       <strong>+</strong>
#                     </span>
#                   </h1>
#                 </div>
#                 """
# 
#                 icon = ipyleaflet.DivIcon(
#                     html=html_for_marker, icon_anchor=[0, 0], icon_size=[0, 0]
#                 )
#                 new_point = ipyleaflet.Marker(
#                     location=kwargs.get("coordinates"), icon=icon
#                 )
# 
#                 # added points are tracked (color/symbol assigned) by the order they are added to the layer_group
#                 # (each point/icon is a layer by itself in ipyleaflet)
#                 self._map_picked_points_layer_group.add_layer(new_point)
#                 print(f"point added {kwargs.get('coordinates')}")
#                 self.icon_color_index += 1
#                 # if icon_color_index>=len(colornames):
#                 #    icon_color_index=0
#             else:
#                 self._last_click = kwargs
# 
#     def _plot_by_satellite(self, ins3xr, point_v, ax, point_xy, map_epsg):
#      
#         try:
#             sat = np.array([x[0] for x in ins3xr["satellite_img1"].values])
#         except:
#             sat = np.array([str(int(x)) for x in ins3xr["satellite_img1"].values])
# 
#         sats = np.unique(sat)
#         sat_plotsym_dict = {
#             "1": "r+",
#             "2": "b+",
#             "4": "y+",
#             "5": "y+",
#             "7": "c+",
#             "8": "g+",
#             "9": "m+",
#         }
# 
#         sat_label_dict = {
#             "1": "Sentinel 1",
#             "2": "Sentinel 2",
#             "4": "Landsat 4",
#             "5": "Landsat 5",
#             "7": "Landsat 7",
#             "8": "Landsat 8",
#             "9": "Landsat 9",
#         }
# 
#         ax.set_xlabel("Date")
#         ax.set_ylabel("Speed (m/yr)")
#         ax.set_title("ITS_LIVE Ice Flow Speed m/yr")
# 
#         max_dt = self.config["max_separation_days"]
#         dt = ins3xr["date_dt"].values
#         # TODO: document this
#         dt = dt.astype(float) * 1.15741e-14
#         if self._control_plot_running_mean_checkbox.value:
#             runmean, ts = self.runningMean(
#                 ins3xr.mid_date[dt < max_dt].values,
#                 point_v[dt < max_dt].values,
#                 5,
#                 30,
#             )
#             ax.plot(
#                 ts,
#                 runmean,
#                 linestyle="-",
#                 color=plt.cm.tab10(self.color_index),
#                 linewidth=2,
#             )
# 
#         for satellite in sats[::-1]:
#             if any(sat == satellite):
#                 ax.plot(
#                     ins3xr["mid_date"][(sat == satellite) & (dt < max_dt)],
#                     point_v[(sat == satellite) & (dt < max_dt)],
#                     sat_plotsym_dict[satellite],
#                     label=sat_label_dict[satellite],
#                 )
# 
#     def _plot_by_points(self, ins3xr, point_v, ax, point_xy, map_epsg):
#         point_label = f"Point ({round(point_xy[0], 2)}, {round(point_xy[1], 2)})"
#         print(point_xy)
# 
#         dt = ins3xr["date_dt"].values
#         # TODO: document this
#         dt = dt.astype(float) * 1.15741e-14
# 
#         max_dt = self.config["max_separation_days"]
#         # set the maximum image-pair time separation (dt) that will be plotted
#         alpha_value = 0.75
#         marker_size = 3
#         if self._control_plot_running_mean_checkbox.value:
#             alpha_value = 0.25
#             marker_size = 2
#             runmean, ts = self.runningMean(
#                 ins3xr.mid_date[dt < max_dt].values,
#                 point_v[dt < max_dt].values,
#                 5,
#                 30,
#             )
#             ax.plot(
#                 ts,
#                 runmean,
#                 linestyle="-",
#                 color=plt.cm.tab10(self.color_index),
#                 linewidth=2,
#             )
#         ax.plot(
#             ins3xr.mid_date[dt < max_dt],
#             point_v[dt < max_dt],
#             linestyle="None",
#             markeredgecolor=plt.cm.tab10(self.color_index),
#             markerfacecolor=plt.cm.tab10(self.color_index),
#             marker="o",
#             alpha=alpha_value,
#             markersize=marker_size,
#             label=point_label,
#         )
# 
#     def plot_point_on_fig(self, point_xy, ax, map_epsg):
# 
#         # pointxy is [x,y] coordinate in mapfig projection (map_epsg below), nax is plot axis for time series plot
#         start = time.time()
#         print(
#             f"fetching timeseries for point x={point_xy[0]:10.2f} y={point_xy[1]:10.2f}",
#             flush=True,
#         )
#         if "plot" in self.config:
#             variable = self.config["plot"]
#         else:
#             variable = "v"
# 
#         ins3xr, ds_velocity_point, point_tilexy = self.get_timeseries(
#             point_xy, map_epsg, variable
#         )  # returns xarray dataset object (used for time axis in plot) and already loaded v time series
#         if ins3xr is not None:
#             # print(ins3xr)
#             if self.config["color_by"] == "satellite":
#                 self._plot_by_satellite(
#                     ins3xr, ds_velocity_point, ax, point_xy, map_epsg
#                 )
#             else:
#                 self._plot_by_points(ins3xr, ds_velocity_point, ax, point_xy, map_epsg)
#             plt.tight_layout()
#             handles, labels = plt.gca().get_legend_handles_labels()
#             by_label = dict(zip(labels, handles))
#             plt.legend(
#                 by_label.values(), by_label.keys(), loc="upper left", fontsize=10
#             )
#             total_time = time.time() - start
#             print(
#                 f"elapsed time: {total_time:10.2f} - {len(ds_velocity_point)/total_time:6.1f} points per second",
#                 flush=True,
#             )
#         self.color_index += 1
# 
#     def plot_time_series(self, *args, **kwargs):
# 
#         # reset plot and color index
#         self.ax.clear()
#         self.ax.set_xlabel("date")
#         self.ax.set_ylabel("speed (m/yr)")
#         self.ax.set_title(
#             "ITS_LIVE Ice Flow Speed m/yr"
#         )
#         self.fig.tight_layout()
#         self.color_index = 0
# 
#         picked_points_latlon = [
#             a.location for a in self._map_picked_points_layer_group.layers
#         ]
#         if len(picked_points_latlon) > 0:
#             print("Plotting...")
#             for lat, lon in picked_points_latlon:
#                 self.plot_point_on_fig([lon, lat], self.ax, "4326")
#             print("done plotting")
#         else:
#             print("no picked points to plot yet - pick some!")
# 
#     def clear_points(self, *args, **kwargs):
#         self.ax.clear()
#         self.color_index = 0
#         self.icon_color_index = 0
#         self._map_picked_points_layer_group.clear_layers()
#         print("all points cleared")
# 
#     def get_zarr_cubes(self):
#         return [(k, v) for k, v in self.open_cubes.items()]
