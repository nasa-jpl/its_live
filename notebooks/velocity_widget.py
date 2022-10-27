# for timing data access
import io
import shutil
import time
import zipfile
from pathlib import Path
from uuid import uuid4

import ipyleaflet
import ipywidgets
import markdown
import numpy as np
import pandas as pd
# to get and use geojson datacube catalog
# for datacube xarray/zarr access
from IPython.display import display
from ipywidgets import HTML, FileUpload, widgets
# for plotting time series
from matplotlib import pyplot as plt

# import itslive datacube tools for working with cloun-based datacubes
from datacube_tools import DATACUBETOOLS as dctools


class ITSLIVE:
    """
    Class to encapsulate ITS_LIVE plotting from zarr in S3
    """

    VELOCITY_ATTRIBUTION = """ \nITS_LIVE velocity mosaic
    (<a href="https://its-live.jpl.nasa.gov">ITS_LIVE</a>) with funding provided by NASA MEaSUREs.\n
    """

    def __init__(self, *args, **kwargs):
        """
        Map widget to plot glacier velocities
        """
        self.dct = (
            dctools()
        )  # initializes geojson catalog and open cubes list for this object

        self.directory_session = uuid4()

        self.ts = []

        self.color_index = 0
        self.icon_color_index = 0
        self._last_click = None
        self.fig, self.fig_h = plt.figure(1), plt.figure(2)
        self.ax, self.ax_h = self.fig.add_subplot(111), self.fig_h.add_subplot(111)

        # self._initialize_widgets()

    def set_config(self, config):
        self.config = config

    def _initialize_widgets(self, projection="global", render_mobile=True):
        self._control_plot_running_mean_checkbox = ipywidgets.Checkbox(
            value=True,
            description="Include running mean",
            disabled=False,
            indent=False,
            tooltip="Plot running mean through each time series",
            layout=ipywidgets.Layout(width="150px"),
        )
        self._control_plot_running_mean_widgcntrl = ipyleaflet.WidgetControl(
            widget=self._control_plot_running_mean_checkbox, position="bottomright"
        )
        self._control_clear_points_button = ipywidgets.Button(
            description="Clear Points", tooltip="clear all picked points"
        )
        self._control_clear_points_button.on_click(self.clear_points)

        self._control_clear_points_button_widgcntrl = ipyleaflet.WidgetControl(
            widget=self._control_clear_points_button, position="bottomright"
        )

        self._control_plot_button = ipywidgets.Button(
            description="Draw Marker", tooltip="click to make plot"
        )
        self._control_plot_button.style.button_color = "lightgreen"
        self._control_plot_button.on_click(self.plot_time_series)
        self._control_plot_button_widgcntrl = ipyleaflet.WidgetControl(
            widget=self._control_plot_button, position="bottomleft"
        )

        self._map_base_layer = ipyleaflet.basemap_to_tiles(
            {
                "url": (
                    "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/"
                    "MapServer/tile/{z}/{y}/{x}.jpg"
                ),
                "attribution": "\nImagery provided by ESRI\n",
                "name": "ESRI basemap",
            }
        )
        self._map_velocity_layer = ipyleaflet.basemap_to_tiles(
            {
                "url": "https://glacierflow.nyc3.digitaloceanspaces.com/webmaps/vel_map/{z}/{x}/{y}.png",
                "attribution": self.VELOCITY_ATTRIBUTION,
                "name": "ITS_LIVE Velocity Mosaic",
            }
        )
        self._map_coastlines_layer = ipyleaflet.basemap_to_tiles(
            {
                "url": "https://gibs.earthdata.nasa.gov/wmts/epsg3857/best/Coastlines_15m/default/GoogleMapsCompatible_Level13/{z}/{y}/{x}.png",
                "attribution": "NASA GIBS Imagery",
                "name": "Coastlines",
            }
        )
        self._map_landmask_layer = ipyleaflet.basemap_to_tiles(
            {
                "url": "https://gibs.earthdata.nasa.gov/wmts/epsg3857/best/OSM_Land_Mask/default/GoogleMapsCompatible_Level9/{z}/{y}/{x}.png",
                "attribution": "NASA GIBS Imagery",
                "name": "Land Mask",
            }
        )

        # OSM_Land_Mask
        # self._map_coastlines_layer.base = True
        self._map_coverage_layer = ipyleaflet.GeoJSON(
            data=self.dct.json_catalog,
            name="ITS_LIVE datacube coverage",
            style={
                "opacity": 0.8,
                "fillOpacity": 0.2,
                "weight": 1,
                "color": "red",
                "cursor": "crosshair",
            },
            hover_style={
                "color": "white",
                "dashArray": "0",
                "fillOpacity": 0.5,
            },
        )
        self.map = ipyleaflet.Map(
            double_click_zoom=False,
            scroll_wheel_zoom=True,
            center=[57.20, -49.43],
            zoom=4
            # layout=ipywidgets.Layout(height="100%", max_height="100%", display="flex")
        )
        self._map_velocity_layer.base = True
        self._map_base_layer.base = True
        self._map_picked_points_layer_group = ipyleaflet.LayerGroup(
            layers=[], name="Selected Points"
        )

        self._map_landmask_layer.opacity = 0.5
        # Populating the map

        self.map.add_layer(self._map_picked_points_layer_group)
        self.map.add_layer(self._map_coastlines_layer)
        self.map.add_layer(self._map_landmask_layer)
        self.map.add_layer(self._map_base_layer)
        self.map.add_layer(self._map_velocity_layer)
        # wms = ipyleaflet.WMSLayer(url="https://integration.glims.org/geoserver/GLIMS/gwc/service",
        #                           name="GLIMS glacier outlines",
        #                           layers="GLIMS:GLIMS_GLACIERS",
        #                           transparent=True,
        #                           opacity=0.33,
        #                           format='image/png')
        # self.map.add_layer(wms)
        self.map.add_control(
            ipyleaflet.MeasureControl(
                position="topleft",
                active_color="orange",
                primary_length_unit="kilometers",
            )
        )
        marker = ipyleaflet.Marker(
            icon=ipyleaflet.AwesomeIcon(
                name="check", marker_color="green", icon_color="darkgreen"
            )
        )
        self.map.add_control(ipyleaflet.FullScreenControl())
        self.map.add_control(ipyleaflet.LayersControl())
        self.map.add_control(
            ipyleaflet.SearchControl(
                position="topleft",
                url="https://nominatim.openstreetmap.org/search?format=json&q={s}",
                zoom=5,
                marker=marker,
            )
        )
        self.map.add_control(ipyleaflet.ScaleControl(position="bottomleft"))
        # self.map.add_control(self._control_plot_running_mean_widgcntrl)
        # self.map.add_control(self._control_clear_points_button_widgcntrl)
        # self.map.add_control(self._control_plot_button_widgcntrl)
        self.map.default_style = {"cursor": "crosshair"}
        self.map.on_interaction(self._handle_map_click)
        self.map.world_copy_jump = True

        self._dates_range = widgets.SelectionRangeSlider(
            options=[i for i in range(546)],
            index=(1, 120),
            continuous_update=False,
            description="Interval (days): ",
            orientation="horizontal",
            layout={"width": "90%", "display": "flex"},
            style={"description_width": "initial"},
        )

        self._variables = widgets.Dropdown(
            options=["v", "v_error", "vx", "vy"],
            description="Variable: ",
            disabled=False,
            value="v",
            layout={"width": "20%", "display": "flex"},
            style={"description_width": "initial"},
        )

        self._plot_type = widgets.Dropdown(
            options=["location", "satellite"],
            description="Plot By: ",
            disabled=False,
            value="location",
            layout={"width": "20%", "display": "flex"},
            style={"description_width": "initial"},
        )

        self._plot_button = widgets.Button(
            description="Plot",
            button_style="primary",
            icon="line-chart",
            style={"description_width": "initial"},
        )

        self._clear_button = widgets.Button(
            description="Clear Points",
            # button_style='warning',
            icon="trash",
            style={"description_width": "initial"},
        )

        self._latitude = widgets.BoundedFloatText(
            value=0.0,
            min=-90.0,
            max=90.0,
            step=0.1,
            description="Lat: ",
            disabled=False,
            style={"description_width": "initial"},
            layout={"width": "20%", "display": "flex"},
        )

        self._longitude = widgets.BoundedFloatText(
            value=0.0,
            min=-180.0,
            max=180.0,
            step=0.1,
            description="Lon: ",
            disabled=False,
            style={"description_width": "initial"},
            layout={"width": "20%", "display": "flex"},
        )

        self._f_upload = FileUpload(
            accept="*.csv",
            description="Import coords",
            multiple=False,  # True to accept multiple files upload else False
        )

        self._add_button = widgets.Button(
            description="Add Point",
            # button_style='info',
            icon="map-marker",
            style={"description_width": "initial"},
        )

        self._include_running_mean = widgets.Checkbox(
            value=False,
            description="Include Running Mean",
            style={"description_width": "initial"},
            disabled=False,
            indent=False,
            tooltip="Plot running mean through each time series",
            layout=widgets.Layout(width="25%"),
        )

        self._export_button = widgets.Button(
            description="Export Data",
            # button_style='info',
            icon="file-export",
            style={"description_width": "initial"},
        )

        self._data_link = widgets.HTML(value="<br>")

        self._plot_button.on_click(self.plot_time_series)
        self._clear_button.on_click(self.clear_points)

        def update_variable(change):
            if change["type"] == "change" and change["name"] == "value":
                self.config["plot"] = self._variables.value
                self.set_config(self.config)
                self.plot_time_series()

        def update_range(change):
            if change["type"] == "change" and change["name"] == "value":
                start, end = change["new"]
                self.config["min_separation_days"] = start
                self.config["max_separation_days"] = end
                self.set_config(self.config)
                self.plot_time_series()

        def update_plottype(change):
            if change["type"] == "change" and change["name"] == "value":
                self.config["color_by"] = self._plot_type.value
                self.set_config(self.config)
                self.plot_time_series()

        def update_mean(change):
            if change["type"] == "change" and change["name"] == "value":
                self.config["running_mean"] = self._include_running_mean.value
                self.set_config(self.config)
                self.plot_time_series()

        def add_point(event):
            if type(event) != widgets.Button:
                uploaded = self._f_upload.value
                file_name = list(uploaded.keys())[0]
                coords = self.import_points(
                    io.StringIO(uploaded[file_name]["content"].decode("utf-8"))
                )
            else:
                coords = (self._latitude.value, self._longitude.value)
                self.add_point(coords)

        def export_ts(event):
            self.export_data()

        self._export_button.on_click(export_ts)

        self._add_button.on_click(add_point)
        self._dates_range.observe(update_range, "value")
        self._plot_type.observe(update_plottype, "value")
        self._variables.observe(update_variable, "value")
        self._include_running_mean.observe(update_mean, "value")

        self._f_upload.observe(add_point, names="value")

        layout = widgets.Layout(
            align_items="stretch",
            display="flex",
            flex_flow="row wrap",
            border="none",
            grid_template_columns="repeat(auto-fit, minmax(720px, 1fr))",
            # grid_template_columns='48% 48%',
            width="99%",
            height="100%",
        )
        self._velocity_controls = widgets.HBox(
            [
                self._variables,
                self._plot_type,
                self._include_running_mean,
            ],
            layout=widgets.Layout(
                justify_content="flex-start",
                flex_flow="row wrap",
            ),
        )
        self._plot_tab = widgets.Tab()
        if render_mobile:
            chart_layout = layout = widgets.Layout(
                min_width="420px",
                max_width="100%",
                style={"description_width": "initial"},
            )
        else:
            chart_layout = layout = widgets.Layout(
                min_width="720px",
                max_width="100%",
                style={"description_width": "initial"},
            )
        self._plot_tab.children = [
            widgets.VBox(
                [
                    self._dates_range,
                    self._velocity_controls,
                    self.fig.canvas,
                    widgets.HBox([self._export_button, self._data_link]),
                ],
                layout=chart_layout,
            ),
            widgets.VBox([self.fig_h.canvas], layout=chart_layout),
        ]
        [
            self._plot_tab.set_title(i, title)
            for i, title in enumerate(["Velocity", "Elevation Change (Antarctica)"])
        ]
        self._plot_tab.style = {"description_width": "initial"}

        html_title = markdown.markdown(
            """<div>
                <h2>
                  <center>
                    <a href="https://its-live.jpl.nasa.gov/"><img align="middle" src="https://its-live-data.s3.amazonaws.com/documentation/ITS_LIVE_logo.png" height="50"/></a>
                  </center>
                </h2>
                <h3>
                  <center>Global Glacier Velocity Point Data Access</center>
                </h3>
            </div>
***
"""
        )
        html_instructions = markdown.markdown(
            """<table><tr>Click and drag on the map to pan the field of view. Select locations by double-clicking on the map then press Plot.
            Once plotted you can change the Variable that is being shown and how the markers are colored using Plot By.
            You can drag individual points after they are placed to relocate them, and then Plot again or Clear markers to start over.
            You can also single-click on the map to populate the Lat and Lon boxes then add a point using the Add Point.
            Lat and Lon can also be edited manually. Hovering your cursor over the plot reveals tools to zoom, pan, and save the figure.
            Importing coordinates: We can use a CSV file with lat, lon values and the tool will place them in the map ready to be plotted.
            The file should be in the following format:
            <br>
            `70.3456,-45.0856`<br>
            `71.0763,-45.0235`<br>
            In order to have a clear plot no more than 20 points is adviced.
            <b>Tip:</b> We can resize the plot by dragging the corner mark!
            Press Export Data to generate comma separated value (.csv) files of the data. Press Download Data to retrieve locally.
            Export Data must be pressed each time new data is requested.
            Data are Version 2 of the ITS_LIVE global glacier velocity dataset that provides up-to-date velocities from Sentinel-1, Sentinel-2, Landsat-8 and Landsat-9 data.
            Version 2 annual mosaics are coming soon, and will be followed by Landsat 7 and improved Landsat 9 velocities.
            Please refer to the <b><a href="https://its-live.jpl.nasa.gov/">project website</a></b> for known issues, citation and other product information."""
        )
        instructions_vid = markdown.markdown(
            """<center>
              <a href="https://www.youtube.com/watch?v=VYKsVvpVbmU" target="_blank">
                <img width="280px" src="https://its-live-data.s3.amazonaws.com/documentation/ITS_LIVE_widget_youtube.jpg">
              </a>
            </center>
            Check out the video tutorial if you're a visual learner"""
        )
        self._title = HTML(
            html_title,
            layout=widgets.Layout(width="100%", display="flex", align_items="stretch"),
        )
        self._instructions = widgets.Accordion(
            children=[widgets.HBox([HTML(html_instructions), HTML(instructions_vid)])],
            selected_index=None,
            layout=widgets.Layout(width="100%", display="flex", align_items="stretch"),
        )
        self._instructions.set_title(0, title="Instructions")

        self.ui = widgets.GridBox(
            [
                widgets.VBox(
                    [
                        widgets.VBox(
                            [self._title, self._instructions],
                            layout=widgets.Layout(width="100%"),
                        ),
                        widgets.VBox(
                            [
                                self.map,
                                widgets.HBox(
                                    [
                                        self._latitude,
                                        self._longitude,
                                        self._add_button,
                                        self._clear_button,
                                        self._f_upload,
                                        self._plot_button,
                                    ],
                                    layout=widgets.Layout(
                                        align_items="flex-start", flex_flow="row wrap"
                                    ),
                                ),
                            ],
                            layout=widgets.Layout(
                                min_width="420px",
                                # display="flex",
                                # height="100%",
                                # max_height="100%",
                                max_width="100%",
                            ),
                        ),
                    ],
                    layout=widgets.Layout(
                        min_width="100%",
                        display="flex",
                        # height="100%",
                        # max_height="100%",
                        max_width="100%",
                    ),
                ),
                widgets.VBox(
                    [
                        self._plot_tab,
                    ],
                    layout=widgets.Layout(
                        min_width="420px",
                        overflow="scroll",
                        max_width="100%",
                        display="flex",
                    ),
                ),
            ],
            layout=layout,
        )
        self.config = {
            "plot": "v",
            "min_separation_days": 5,
            "max_separation_days": 90,
            "color_by": "location",
            "verbose": False,
            "running_mean": False,
            "coords": {"latitude": self._latitude, "longitude": self._longitude},
            "data_link": self._data_link,
            "title": None,
            "instructions": None,
        }
        self.set_config(self.config)

    def display(self, render_sidecar=False, mobile=True):
        if render_sidecar:
            from sidecar import Sidecar

            if not hasattr(self, "sidecar"):
                self.sidecar = Sidecar(title="Map Widget")
            self.sidecar.clear_output()
            with self.sidecar:
                self._initialize_widgets(render_mobile=mobile)
                self.set_config(self.config)
                self.map.default_style = {"cursor": "crosshair"}
                display(self.ui)
        else:
            self._initialize_widgets(render_mobile=mobile)
            self.set_config(self.config)
            self.map.default_style = {"cursor": "crosshair"}
            display(self.ui)

    # running mean
    def runningMean(
        self,
        mid_dates,
        variable,
        minpts,
        tFreq,
    ):
        """
        mid_dates: center dates of `variable` data [datetime64]
        variable: data to be average
        minpts: minimum number of points needed for a valid value, else filled with nan
        tFreq: the spacing between centered averages in Days, default window size = tFreq*2
        """
        tsmin = pd.Timestamp(np.min(mid_dates))
        tsmax = pd.Timestamp(np.max(mid_dates))
        ts = pd.date_range(start=tsmin, end=tsmax, freq=f"{tFreq}D")
        ts = pd.to_datetime(ts).values
        idx0 = ~np.isnan(variable)
        runmean = np.empty([len(ts) - 1, 1])
        runmean[:] = np.nan
        tsmean = ts[0:-1]

        t_np = mid_dates.astype(np.int64)

        for i in range(len(ts) - 1):
            idx = (
                (mid_dates >= (ts[i] - np.timedelta64(int(tFreq / 2), "D")))
                & (mid_dates < (ts[i + 1] + np.timedelta64(int(tFreq / 2), "D")))
                & idx0
            )
            if sum(idx) >= minpts:
                runmean[i] = np.mean(variable[idx])
                tsmean[i] = np.mean(t_np[idx])

        tsmean = pd.to_datetime(tsmean).values
        return (runmean, tsmean)

    def import_points(self, content):
        df = pd.read_csv(content, usecols=[0, 1], names=["lat", "lon"], header=None)
        last_points = None
        for row in df.itertuples(index=False):
            try:
                if getattr(row, "lat") and getattr(row, "lon"):
                    self.add_point((row.lat, row.lon))
                    last_points = (row.lat, row.lon)
            except Exception:
                print(Exception)
        self.map.fit_bounds(
            [
                [last_points[0] - 5, last_points[1] - 5],
                [last_points[0] + 5, last_points[1] + 5],
            ]
        )
        self.map.zoom = 5

    def add_point(self, point):
        color = plt.cm.tab10(self.icon_color_index)
        if self.config["verbose"]:
            print(self.icon_color_index, color)
        html_for_marker = f"""
        <div>
            <h1 style="position: absolute;left: -0.2em; top: -2.5rem; font-size: 2rem;">
            <span style="color: rgba({color[0]*100}%,{color[1]*100}%,{color[2]*100}%, {color[3]});
                width: 2rem;height: 2rem; display: block;position: relative;transform: rotate(45deg);">
                <strong>+</strong>
            </span>
            </h1>
        </div>
        """

        icon = ipyleaflet.DivIcon(
            html=html_for_marker, icon_anchor=[0, 0], icon_size=[0, 0]
        )
        new_point = ipyleaflet.Marker(location=point, icon=icon)

        # added points are tracked (color/symbol assigned) by the order they are added to the layer_group
        # (each point/icon is a layer by itself in ipyleaflet)
        self._map_picked_points_layer_group.add_layer(new_point)

        if self.config["verbose"]:
            print(f"point added {point}")
        self.icon_color_index += 1

    def _handle_map_click(self, **kwargs):
        if kwargs.get("type") == "click":
            coords = kwargs.get("coordinates")
            # NOTE this is the work around for the double click issue discussed above!
            # Only acknoledge the click when it is registered the second time at the same place!
            if self.config["coords"] is not None:
                print(kwargs.get("coordinates"))
                self.config["coords"]["latitude"].value = round(coords[0], 2)
                self.config["coords"]["longitude"].value = round(coords[1], 2)
            if self._last_click and (
                kwargs.get("coordinates") == self._last_click.get("coordinates")
            ):
                self.add_point(coords)
            else:
                print(kwargs.get("type"))
                self._last_click = kwargs

    def _plot_by_satellite(self, ins3xr, point_v, point_xy, map_epsg):

        try:
            sat = np.array([x[0] for x in ins3xr["satellite_img1"].values])
        except Exception:
            sat = np.array([str(int(x)) for x in ins3xr["satellite_img1"].values])

        sats = np.unique(sat)
        sat_plotsym_dict = {
            "1": "r+",
            "2": "bo",
            "4": "y+",
            "5": "y+",
            "7": "c+",
            "8": "g*",
            "9": "m^",
        }

        sat_label_dict = {
            "1": "Sentinel 1",
            "2": "Sentinel 2",
            "4": "Landsat 4",
            "5": "Landsat 5",
            "7": "Landsat 7",
            "8": "Landsat 8",
            "9": "Landsat 9",
        }

        # self.ax.set_xlabel("Date")
        # self.ax.set_ylabel("Speed (m/yr)")
        # self.ax.set_title("ITS_LIVE Ice Flow Speed m/yr")

        max_dt = self.config["max_separation_days"]
        min_dt = self.config["min_separation_days"]
        dt = ins3xr["date_dt"].values
        # TODO: document this
        dt = dt.astype(float) * 1.15741e-14
        if "running_mean" in self.config and self.config["running_mean"]:
            runmean, ts = self.runningMean(
                ins3xr.mid_date[(dt >= min_dt) & (dt <= max_dt)].values,
                point_v[(dt >= min_dt) & (dt <= max_dt)].values,
                5,
                30,
            )
            self.ax.plot(
                ts,
                runmean,
                linestyle="-",
                color=plt.cm.tab10(self.color_index),
                linewidth=2,
            )

        for satellite in sats[::-1]:
            if any(sat == satellite):
                self.ax.plot(
                    ins3xr["mid_date"][
                        (sat == satellite) & (dt >= min_dt) & (dt <= max_dt)
                    ],
                    point_v[(sat == satellite) & (dt >= min_dt) & (dt <= max_dt)],
                    sat_plotsym_dict[satellite],
                    markersize=3,
                    label=sat_label_dict[satellite],
                )

    def _plot_by_points(self, ins3xr, point_v, point_xy, map_epsg):
        point_label = f"Lat: {round(point_xy[1], 4)}, Lon: {round(point_xy[0], 4)}"
        if self.config["verbose"]:
            print(point_xy)

        dt = ins3xr["date_dt"].values
        # TODO: document this
        dt = dt.astype(float) * 1.15741e-14

        max_dt = self.config["max_separation_days"]
        min_dt = self.config["min_separation_days"]
        # set the maximum image-pair time separation (dt) that will be plotted
        alpha_value = 0.75
        marker_size = 3
        if "running_mean" in self.config and self.config["running_mean"]:
            alpha_value = 0.25
            marker_size = 2
            runmean, ts = self.runningMean(
                ins3xr.mid_date[(dt >= min_dt) & (dt <= max_dt)].values,
                point_v[(dt >= min_dt) & (dt <= max_dt)].values,
                5,
                30,
            )
            self.ax.plot(
                ts,
                runmean,
                linestyle="-",
                color=plt.cm.tab10(self.color_index),
                linewidth=2,
            )
        self.ax.plot(
            ins3xr.mid_date[(dt >= min_dt) & (dt <= max_dt)],
            point_v[(dt >= min_dt) & (dt <= max_dt)],
            linestyle="None",
            markeredgecolor=plt.cm.tab10(self.color_index),
            markerfacecolor=plt.cm.tab10(self.color_index),
            marker="o",
            alpha=alpha_value,
            markersize=marker_size,
            label=point_label,
        )

    def plot_elevation(self, h, coords):
        point_label = f"Lat: {round(coords[1], 4)}, Lon: {round(coords[0], 4)}"
        alpha_value = 0.75
        marker_size = 3
        self.ax_h.plot(
            h,
            linestyle="None",
            markeredgecolor=plt.cm.tab10(self.color_index),
            markerfacecolor=plt.cm.tab10(self.color_index),
            marker="o",
            alpha=alpha_value,
            markersize=marker_size,
            label=point_label,
        )
        self.ax_h.set_xlabel("time")
        self.ax_h.set_ylabel("elevation change (meters)")

    def plot_point_on_fig(self, point_xy, map_epsg):

        # pointxy is [x,y] coordinate in mapfig projection (map_epsg below), nax is plot axis for time series plot
        start = time.time()
        if self.config["verbose"]:
            print(
                f"fetching timeseries for point x={point_xy[0]:10.2f} y={point_xy[1]:10.2f}",
                flush=True,
            )
        if "plot" in self.config:
            variable = self.config["plot"]
        else:
            variable = "v"

        ins3xr, ds_point, point_tilexy = self.dct.get_timeseries_at_point(
            point_xy, map_epsg, variables=[variable]
        )
        if ins3xr is not None:
            export = ins3xr[
                [
                    "v",
                    "v_error",
                    "vx",
                    "vx_error",
                    "vy",
                    "vy_error",
                    "date_dt",
                    "satellite_img1",
                    "mission_img1",
                ]
            ].sel(x=point_tilexy[0], y=point_tilexy[1], method="nearest")

            self.ts.append((export, point_xy))
            ds_velocity_point = ds_point[variable]
            # dct.get_timeseries_at_point returns dataset, extract dataArray for variable from it for plotting
            # returns xarray dataset object (used for time axis in plot) and already loaded v time series

            def legend_without_duplicate_labels(ax):
                handles, labels = ax.get_legend_handles_labels()
                unique = [
                    (h, l)
                    for i, (h, l) in enumerate(zip(handles, labels))
                    if l not in labels[:i]
                ]
                ax.legend(*zip(*unique), loc="upper right", fontsize=9)

            if point_xy[1] < -60:
                ts = self.dct.load_elevation_timeseries(point_xy[0], point_xy[1])
                if ts is not None:
                    self.plot_elevation(ts, point_xy)
                    if self.ax_h.get_legend() is not None:
                        self.ax_h.get_legend().remove()
                    legend_without_duplicate_labels(self.ax_h)

            if self.config["color_by"] == "satellite":
                self._plot_by_satellite(ins3xr, ds_velocity_point, point_xy, map_epsg)
            else:
                self._plot_by_points(ins3xr, ds_velocity_point, point_xy, map_epsg)

            if self.ax.get_legend() is not None:
                self.ax.get_legend().remove()

            legend_without_duplicate_labels(self.ax)

            total_time = time.time() - start

            if self.config["verbose"]:
                print(
                    f"elapsed time: {total_time:10.2f} - {len(ds_velocity_point)/total_time:6.1f} points per second",
                    flush=True,
                )
            self.color_index += 1

    def export_data(self, *args, **kwargs):
        dir_name = uuid4()
        directory = Path(f"data/{dir_name}/series")
        directory.mkdir(parents=True, exist_ok=True)

        for time_series in self.ts:
            # time_series[0].load()
            lat = round(time_series[1][1], 4)
            lon = round(time_series[1][0], 4)
            df = time_series[0].to_dataframe()
            df["x"] = lon
            df["y"] = lat
            df = df.rename(
                columns={
                    "x": "lon",
                    "y": "lat",
                    "satellite_img1": "satellite",
                    "mission_img1": "mission",
                    "v": "v [m/yr]",
                    "v_error": "v_error [m/yr]",
                    "vx": "vx [m/yr]",
                    "vx_error": "vx_error [m/yr]",
                    "vy": "vy [m/yr]",
                    "vy_error": "vy_error [m/yr]",
                }
            )
            ts = df.dropna()
            ts["epsg"] = time_series[0].attrs["projection"]
            ts["date_dt [days]"] = ts["date_dt"].dt.days
            file_name = f"LAT{lat}--LON{lon}.csv"
            ts.to_csv(
                f"data/{dir_name}/series/{file_name}",
                columns=[
                    "lat",
                    "lon",
                    "v [m/yr]",
                    "v_error [m/yr]",
                    "vx [m/yr]",
                    "vx_error [m/yr]",
                    "vy [m/yr]",
                    "vy_error [m/yr]",
                    "date_dt [days]",
                    "mission",
                    "satellite",
                    "epsg",
                ],
            )

        with zipfile.ZipFile(
            f"data/{dir_name}/itslive-data.zip", "w", zipfile.ZIP_DEFLATED
        ) as zip_file:
            for entry in directory.rglob("*"):
                zip_file.write(entry, entry.relative_to(directory))

        shutil.rmtree(f"data/{dir_name}/series")
        if self.config["data_link"]:
            self.config[
                "data_link"
            ].value = f"""
            <a target="_blank" href="data/{dir_name}/itslive-data.zip" >
                <div class="jupyter-button mod-warning">Download Data</div>
            </a>
            """

    def plot_time_series(self, *args, **kwargs):

        # reset plot and color index
        self.ax.clear()
        self.ax_h.clear()

        self.ax.set_xlabel("date")
        self.ax.set_ylabel("speed (m/yr)")
        self.fig.tight_layout()
        self.fig_h.tight_layout()
        self.color_index = 0
        self.ts = []

        picked_points_latlon = [
            a.location for a in self._map_picked_points_layer_group.layers
        ]
        if len(picked_points_latlon) > 0:
            self.ax.set_title("Plotting...")
            self.fig.canvas.draw()
            self._control_plot_button.disabled = True
            if self.config["verbose"]:
                print("Plotting...")
            for lat, lon in picked_points_latlon:
                self.plot_point_on_fig([lon, lat], "4326")
            if self.config["verbose"]:
                print("done plotting")
            # plt.get_current_fig_manager().canvas.set_window_title("")
            self.ax.set_title("ITS_LIVE Ice Flow Speed m/yr")
            self.ax_h.set_title("ITS_LIVE Elevation Change (m)")

            self.fig.tight_layout()
            self.fig.canvas.draw()
            self.fig_h.tight_layout()
            self.fig_h.canvas.draw()

            self._control_plot_button.disabled = False
        else:
            print("no picked points to plot yet - pick some!")

    def clear_points(self, *args, **kwargs):
        self.ax.clear()
        self.color_index = 0
        self.icon_color_index = 0
        self._map_picked_points_layer_group.clear_layers()
        print("all points cleared")
