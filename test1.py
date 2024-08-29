# -*- coding: utf-8 -*-
import platform, sys, os
import requests
"""### Define ENV Variables for Jupyter Instance"""

# Commented out IPython magic to ensure Python compatibility.
# use default GMTSAR installation path
PATH = os.environ['PATH']
if 'GMTSAR' not in PATH:
    PATH += ':/usr/local/GMTSAR/bin/'
    os.environ['PATH'] = PATH

"""## Load and Setup Python Modules"""

import xarray as xr
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import box, shape, Point, Polygon, MultiLineString, LineString
import shapely.errors
from dask.distributed import Client
import dask
import warnings
from ipyleaflet import Map, TileLayer, LayersControl, GeoJSON, Marker
warnings.filterwarnings('ignore')
# plotting modules
import pyvista as pv
pv.set_plot_theme("document")
import panel
panel.extension('vtk')
from contextlib import contextmanager
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource
import matplotlib.colors as mcolors
import json
@contextmanager
def mpl_settings(settings):
    original_settings = {k: plt.rcParams[k] for k in settings}
    plt.rcParams.update(settings)
    yield
    plt.rcParams.update(original_settings)
plt.rcParams['figure.figsize'] = [12, 4]
plt.rcParams['figure.dpi'] = 100
plt.rcParams['figure.titlesize'] = 24
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12


from pygmtsar import S1, Stack, tqdm_dask, ASF, Tiles

SCENES_FILE = 'scenes.txt'
SUBSWATH_FILE = 'subswath.txt'
AOI_FILE = 'aoi.json'
CONFIG_FILE = 'config.txt'

def load_config(filename):
    config = {}
    with open(filename, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            name, value = line.strip().split('=',1)
            config[name.strip()] = value.strip()
    return config

with open(SCENES_FILE, 'r') as file:
    SCENES = [line.strip() for line in file if line.strip()]

with open(SUBSWATH_FILE, 'r') as file:
    SUBSWATH = int(file.read().strip())

config = load_config(CONFIG_FILE)
POLARIZATION = config['POLARIZATION']
REFERENCE = config['REFERENCE']
WORKDIR = config['WORKDIR']
DATADIR = config['DATADIR']
BASEDAYS = int(config['BASEDAYS'])
BASEMETERS = int(config['BASEMETERS'])
marker_lon = float(config['marker_lon'])
marker_lat = float(config['marker_lat'])
center_lon = float(config['center_lon'])
center_lat = float(config['center_lat'])
marker2_lon = float(config['marker2_lon'])
marker2_lat = float(config['marker2_lat'])

# define DEM filename inside data directory
DEM = f'{DATADIR}/dem.nc'
LANDMASK = f'{DATADIR}/landmask.nc'
def reverse_geocode(lat, lon):
    """使用 Nominatim API 進行地理編碼來獲取地名"""
    url = f'https://nominatim.openstreetmap.org/reverse?format=json&lat={lat}&lon={lon}&zoom=10'
    response = requests.get(url)
    data = response.json()
    address = data.get('address', {})
    return address.get('city', '') or address.get('town', '') or address.get('village', '') or "未知地名"

# 使用 Nominatim API 獲取地名
marker1_name = config['marker1_name']
marker2_name = config['marker2_name']

def main():
    
    # Set these variables to None and you will be prompted to enter your username and password below.
    asf_username = config.get('asf_username')
    asf_password = config.get('asf_password')
    
    # Set these variables to None and you will be prompted to enter your username and password below.
    asf = ASF(asf_username, asf_password)
    # Optimized scene downloading from ASF - only the required subswaths and polarizations.
    print(asf.download_scenes(DATADIR, SCENES, SUBSWATH))
    
    # scan the data directory for SLC scenes and download missed orbits
    S1.download_orbits(DATADIR, S1.scan_slc(DATADIR))

    with open(AOI_FILE, 'r', encoding='utf-8') as file:
        geojson_data = json.load(file)
        AOI = gpd.GeoDataFrame.from_features(geojson_data['features'])
    
    # download Copernicus Global DEM 1 arc-second
    Tiles().download_dem(AOI, filename=DEM)
    Tiles().download_landmask(AOI, filename=LANDMASK).plot.imshow(cmap='binary')
    
    # simple Dask initialization
    if 'client' in globals():
        client.close()
    client = Client()
    client
    
    """## Init
    
    Search recursively for measurement (.tiff) and annotation (.xml) and orbit (.EOF) files in the DATA directory. It can be directory with full unzipped scenes (.SAFE) subdirectories or just a directory with the list of pairs of required .tiff and .xml files (maybe pre-filtered for orbit, polarization and subswath to save disk space). If orbit files and DEM are missed these will be downloaded automatically below.
    """
    
    scenes = S1.scan_slc(DATADIR)
    scenes
    
    sbas = Stack(WORKDIR, drop_if_exists=True).set_scenes(scenes).set_reference(REFERENCE)
    sbas.to_dataframe()
    
    sbas.plot_scenes()
    
    """### Load DEM
    
    The function below loads DEM from file or Xarray variable and converts heights to ellipsoidal model using EGM96 grid.
    """
    sbas.load_dem(DEM, AOI)
    sbas.plot_scenes()
    plt.savefig('Estimated Scene Locations.jpg')
    
    """### Load LandMask"""
    sbas.load_landmask(LANDMASK)
    sbas.plot_scenes(AOI=AOI, dem=sbas.get_dem().where(sbas.get_landmask()), caption='Sentinel1 Landmasked Frame on DEM')
    plt.savefig('Sentinel1 Landmasked Frame on DEM.jpg')
    """## Align a Stack of Images"""
    
    sbas.compute_align()
    
    """## SBAS Baseline"""
    
    baseline_pairs = sbas.sbas_pairs(days=BASEDAYS, meters=BASEMETERS)
    baseline_pairs
    
    with mpl_settings({'figure.dpi': 150}):
        sbas.plot_baseline(baseline_pairs)
    plt.savefig('Baseline.jpg')

    
    """## Geocoding"""
    
    # use default 60m coordinates grid
    sbas.compute_geocode()
    
    """### DEM in Radar Coordinates
    
    The grids are NetCDF files processing as xarray DataArrays.
    """
    
    sbas.plot_topo()
    plt.savefig('Topography on WGS84 ellipsoid, [m].jpg')
    
    """## Interferograms
    
    Define a single interferogram or a SBAS series. Make direct and reverse interferograms (from past to future or from future to past).
    
    Decimation is useful to save disk space. Geocoding results are always produced on the provided DEM grid so the output grid and resolution are the same to the DEM. By this way, ascending and descending orbit results are always defined on the same grid by design. An internal processing cell is about 30 x 30 meters size and for default output 60m resolution (like to GMTSAR and GAMMA software) decimation 2x2 is reasonable. For the default wavelength=200 for Gaussian filter 1/4 of wavelength is approximately equal to ~60 meters and better resolution is mostly useless (while it can be used for small objects detection). For wavelength=400 meters use 90m DEM resolution with decimation 4x4.
    
    The grids are NetCDF files processing as xarray DataArrays.
    """
    
    pairs = baseline_pairs[['ref', 'rep']]
    pairs
    # load Sentinel-1 data
    data = sbas.open_data()
    
    # Gaussian filtering 400m cut-off wavelength with multilooking 1x4 on Sentinel-1 intensity
    intensity = sbas.multilooking(np.square(np.abs(data)), wavelength=400, coarsen=(1,4))
    
    phase = sbas.multilooking(sbas.phasediff(pairs), wavelength=400, coarsen=(1,4))
    
    corr = sbas.correlation(phase, intensity)
    
    # Goldstein filter expects square grid cells produced using multilooking
    intf_filt = sbas.interferogram(sbas.goldstein(phase, corr, 32))
    
    # use default 60m resolution
    decimator = sbas.decimator()
    
    # compute together because correlation depends on phase, and filtered phase depends on correlation.
    tqdm_dask(result := dask.persist(decimator(corr), decimator(intf_filt)), desc='Compute Phase and Correlation')
    # unpack results
    corr60m, intf60m = result
    #11954/11954 [02:30<00:00, 17.81it/s]

    sbas.plot_interferograms(intf60m, cols=3, size=3, caption='Phase, [rad]')
    plt.savefig('Phase, [rad].jpg')
    
    tqdm_dask(Phase_ll := sbas.ra2ll(intf60m).persist(), desc='Geocoding')
    
    sbas.plot_phases(Phase_ll, cols=3, size=3, caption='Phase in Geographic Coordinates, [rad]', quantile=[0.01, 0.99])
    plt.savefig('Phase in Geographic Coordinates, [rad].jpg')
    
    sbas.plot_correlations(corr60m, cols=3, size=3, caption='Correlation')
    plt.savefig('Correlation.jpg')

    """## Unwrapping
    
    Unwrapping process requires a lot of RAM and that's really RAM consuming when a lot of parallel proccesses running togeter. To limit the parallel processing tasks apply argument "n_jobs". The default value n_jobs=-1 means all the processor cores van be used. Also, use interferogram decimation above to produce smaller interferograms. And in addition a custom SNAPHU configuration can reduce RAM usage as explained below.
    
    Attention: in case of crash on MacOS Apple Silicon run Jupyter as
    
    OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES no_proxy='*' jupyter notebook
    """
    
    CORRLIMIT = 0.075
    tqdm_dask(unwrap := sbas.unwrap_snaphu(intf60m, corr60m.where(corr60m>=CORRLIMIT)).persist(),
              desc='SNAPHU Unwrapping')
    
    sbas.plot_phases(unwrap.phase, cols=3, size=3, caption='Unwrapped Phase, [rad]', quantile=[0.01, 0.99])
    plt.savefig('Unwrapped Phase, [rad].jpg')
    
    tqdm_dask(unwrap_ll := sbas.ra2ll(unwrap.phase).persist(), desc='Geocoding')
    
    sbas.plot_phases(unwrap_ll, cols=3, size=3, caption='Unwrapped Phase in Geographic Coordinates, [rad]', quantile=[0.01, 0.99])
    plt.savefig('Unwrapped Phase in Geographic Coordinates, [rad].jpg')

    
    """### Detrend Unwrapped Phase
    
    Remove trend and apply gaussian filter to fix ionospheric effects and solid Earh's tides.
    """
    
    tqdm_dask(detrend := (unwrap.phase - sbas.gaussian(unwrap.phase, wavelength=60000)).persist(), desc='Detrending')
    
    sbas.plot_phases(detrend, cols=3, size=3, caption='Detrended Unwrapped Phase, [rad]', quantile=[0.01, 0.99])
    plt.savefig('Detrended Unwrapped Phase, [rad].jpg')
    
    tqdm_dask(detrend_ll := sbas.ra2ll(detrend).persist(), desc='Geocoding')
    
    sbas.plot_phases(detrend_ll, cols=3, size=3, caption='Detrended Unwrapped Phase, [rad]', quantile=[0.01, 0.99])
    plt.savefig('Detrended Unwrapped Phase in Geographic Coordinates, [rad].jpg')
    
    """### Calculate Displacement Using Coherence-Weighted Least-Squares Solution"""

    # calculate phase displacement in radians and convert to LOS displacement in millimeter
    tqdm_dask(disp := sbas.los_displacement_mm(sbas.lstsq(detrend, corr60m)).persist(), desc='SBAS Computing')
    
    # clean 1st zero-filled displacement map for better visualization
    disp[0] = np.nan
    
    # Plot the cumulative LOS displacement for the entire dataset
    sbas.plot_displacements(disp, cols=3, size=3, caption='Cumulative LOS Displacement, [mm]', quantile=[0.01, 0.99])
    plt.savefig('Cumulative LOS Displacement, [mm].jpg')
    
    # Geocode the full interferogram grid and crop the valid area only (without restricting to a specific AOI)
    tqdm_dask(disp_ll := sbas.cropna(sbas.ra2ll(disp)).persist(), desc='SBAS Computing')
    
    # Plot the geocoded displacement for the entire dataset
    sbas.plot_displacements(disp_ll, cols=3, size=3, caption='Cumulative LOS Displacement in Geographic Coordinates, [mm]', quantile=[0.01, 0.99])
    plt.savefig('Cumulative LOS Displacement Geographic Coordinates, [mm].jpg')
    # 計算 SBAS LOS 速度 (mm/年)
    velocity_sbas = sbas.velocity(disp_ll)
    
    # 創建圖像
    fig, ax = plt.subplots(figsize=(6, 4), dpi=300)
    # 計算速度數據的百分位以設置顏色映射的範圍
    zmin, zmax = np.nanquantile(velocity_sbas, [0.01, 0.99])
    zminmax = max(abs(zmin), zmax)
    # 顯示速度圖像，使用 'turbo' 色圖，並設定顏色範圍
    velocity_sbas.plot.imshow(cmap='turbo', vmin=-zminmax, vmax=zminmax, ax=ax)
    
    ax.plot(marker_lon, marker_lat, 'ro', markersize=8)  # 'ro' 為紅色標記
    
    # 設置標題
    ax.set_title('Velocity, mm/year', fontsize=16)
    plt.suptitle('SBAS LOS Velocity', fontsize=18)
    # 自動調整佈局
    plt.tight_layout()
    # 保存圖像
    plt.savefig('SBAS LOS Velocity.jpg')

    ## 2D Interactive Map
    # prepare and decimate data
    velocity_stacked = sbas.velocity(disp.stack(stack=['y','x'])).rename('velocity')
    df = sbas.ra2ll(velocity_stacked).to_dataframe().dropna()
    gdf = gpd.GeoDataFrame(df[['lat','lon','velocity']], geometry=gpd.points_from_xy(df.lon, df.lat), crs=4326)
    gdf
    
    # specify pixel boundaries in geo coordinates
    def point_to_rectangle(row):
        #print (row)
        import shapely
        return shapely.geometry.Polygon([
            (row._lon0, row._lat0),
            (row._lon1, row._lat1),
            (row._lon2, row._lat2),
            (row._lon3, row._lat3)
        ])

    dycell = np.array([-0.5, -0.5, 0.5, 0.5])
    dxcell = np.array([-0.5, 0.5, 0.5, -0.5])
    # use the map pixel spacing
    cellsize = (4, 16)
    for idx, dydx in enumerate(zip(dycell*cellsize[0], dxcell*cellsize[1])):
        index = pd.MultiIndex.from_tuples(
            [(y + dydx[0], x + dydx[1]) for y, x in gdf.index],
            names=gdf.index.names
        )
        coords = xr.Dataset(coords={'stack': index})
        coords_dydx = sbas.ra2ll(coords).to_dataframe()[['lat','lon']]
        gdf[f'_lat{idx}'] = coords_dydx.lat.values
        gdf[f'_lon{idx}'] = coords_dydx.lon.values
    gdf['geometry'] = gdf.apply(lambda row: point_to_rectangle(row), axis=1)
    for col in gdf.columns[gdf.columns.str.contains('_')]:
        del gdf[col]
    gdf
    
    import shapely
    from shapely.validation import make_valid
#
#    # 檢查並修復無效的幾何圖形----這邊後面容易出錯
    def fix_degenerate(geom):
        try:
            return make_valid(geom)
        except shapely.errors.GEOSException:
            return None

    gdf['geometry'] = gdf['geometry'].apply(fix_degenerate)

    
    # 使用分組對數據進行處理
    gdf['lat_group'] = (gdf['lat'] // 0.002).astype(int)
    gdf['lon_group'] = (gdf['lon'] // 0.002).astype(int)

    # 合併幾何圖並計算平均速度、緯度和經度
    gdf = gdf.groupby(['lat_group', 'lon_group']).agg({
        'lat': 'mean',
        'lon': 'mean',
        'velocity': 'mean',
        'geometry': lambda x: shapely.ops.unary_union(x)
    }).reset_index()

    # 刪除用於分組的 lat_group 和 lon_group 列
    gdf = gdf.drop(columns=['lat_group', 'lon_group'])

    gdf

    # convert the GeoDataFrame to a GeoJSON-like Python dictionary
    geojson_dict = {
        "type": "FeatureCollection",
        "features": []
    }
    
     # convert Shapely geometry to GeoJSON format
    for _, row in gdf.iterrows():
        feature = {
            "type": "Feature",
            "geometry": shapely.geometry.mapping(row['geometry']),
            "properties": {
                "lat": row['lat'],
                "lon": row['lon'],
                "velocity": row['velocity']
            }
        }
        geojson_dict["features"].append(feature)

    # write the dictionary to a GeoJSON file
    with open('GeoJSON_file.geojson', 'w') as f:
        json.dump(geojson_dict, f, indent=2)
    # load the GeoJSON from the file
    with open('GeoJSON_file.geojson', 'r') as f:
        geojson = json.load(f)
    print ('Pixels loaded:', len(geojson['features']))
    import folium
    import matplotlib.colors as mcolors

    # Load the landmask file
    
    landmask_ds = xr.open_dataset(LANDMASK)

    # Extract landmask data, assuming z > 0 indicates land
    landmask = landmask_ds['z'].values > 0

    # Define a colormap and normalization for the velocity
    colormap = plt.get_cmap('turbo')
    norm = plt.Normalize(vmin=-60, vmax=60)  # Assuming velocity range is -60 to 60 mm/year

    def velocity_to_color(velocity, limits=[-60, 60]):
        """Convert velocity to a color from the colormap."""
        normalized = (velocity - limits[0]) / (limits[1] - limits[0])
        return mcolors.to_hex(colormap(normalized))

    def is_land(lat, lon, landmask, landmask_ds):
        """Check if the given latitude and longitude are on land according to the landmask."""
        lat_min = landmask_ds.lat.min().item()  # Convert DataArray to scalar
        lat_max = landmask_ds.lat.max().item()
        lon_min = landmask_ds.lon.min().item()
        lon_max = landmask_ds.lon.max().item()
        
        lat = np.mean(lat) if isinstance(lat, (list, np.ndarray)) else lat
        lon = np.mean(lon) if isinstance(lon, (list, np.ndarray)) else lon
        
        lat_idx = int((lat - lat_min) / (lat_max - lat_min) * (landmask.shape[0] - 1))
        lon_idx = int((lon - lon_min) / (lon_max - lon_min) * (landmask.shape[1] - 1))
        
        if lat_idx < 0 or lat_idx >= landmask.shape[0] or lon_idx < 0 or lon_idx >= landmask.shape[1]:
            return False
        
        return landmask[lat_idx, lon_idx]

    def extract_lat_lon(coords):
        """Extract latitude and longitude from GeoJSON coordinates."""
        if not isinstance(coords[0], list):
            return coords[1], coords[0]
        else:
            flat_coords = []
            def flatten(coords):
                for item in coords:
                    if isinstance(item[0], list):
                        flatten(item)
                    else:
                        flat_coords.append(item)
            flatten(coords)
            flat_coords = np.array(flat_coords)
            center_lat = np.mean(flat_coords[:, 1])
            center_lon = np.mean(flat_coords[:, 0])
            return center_lat, center_lon

    # Filter GeoJSON features to retain only those on land
    filtered_features = []
    for feature in geojson['features']:
        geometry = feature.get('geometry', {})
        if 'coordinates' not in geometry or not geometry['coordinates']:
            continue
        
        coords = geometry['coordinates']
        lat, lon = extract_lat_lon(coords)
        if is_land(lat, lon, landmask, landmask_ds):
            filtered_features.append(feature)

    # Update GeoJSON data
    geojson['features'] = filtered_features
    with open('GeoJSON_file.geojson', 'w') as f:
        json.dump(geojson, f)

    # Simplify the geometries using Geopandas, preserving 95% of the shape
    gdf = gpd.GeoDataFrame.from_features(geojson["features"])
    gdf["geometry"] = gdf["geometry"].simplify(tolerance=0.0003, preserve_topology=True)
    # Update the GeoJSON data
    geojson = json.loads(gdf.to_json())
    
    colormap = plt.get_cmap('turbo')
    norm = plt.Normalize(vmin=-60, vmax=60)  # Assuming velocity range is -60 to 60 mm/year

    # Apply color to each feature
    for feature in geojson['features']:
        color = velocity_to_color(feature['properties']['velocity'])
        feature['properties']['color'] = color

    # Save the simplified GeoJSON file
    with open('GeoJSON_file_simplified.geojson', 'w') as f:
        json.dump(geojson, f)

    # Generate the color bar matching the InSAR colormap
    colorbar_html = ''
    num_colors = 256  # Use 256 colors
    for i in range(num_colors):
        color = mcolors.to_hex(colormap(i / (num_colors - 1)))
        colorbar_html += f'<div style="flex: 1; background-color: {color};"></div>'

    # Generate the HTML file
    html_content = f'''
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>InSAR Velocity Map with Azure Maps</title>
        <script src="https://atlas.microsoft.com/sdk/javascript/mapcontrol/2/atlas.min.js"></script>
        <link rel="stylesheet" href="https://atlas.microsoft.com/sdk/javascript/mapcontrol/2/atlas.min.css">
        <style>
            body {{ margin: 0; padding: 0; }}
            #map {{ width: 100%; height: 100vh; }}
            #popup {{ background-color: white; border: 1px solid black; padding: 5px; position: absolute; display: none; }}
            #colorbar {{
                position: absolute;
                bottom: 20px;
                left: 20px;
                width: 300px;  /* Adjust width for a wider colorbar */
                height: 20px;
                display: flex;
            }}
            #colorbar-labels {{
                position: absolute;
                bottom: 0;
                left: 20px;
                display: flex;
                justify-content: space-between;
                width: 300px;
                font-size: 12px;
                color: black;
            }}
            #layerToggle {{
                position: absolute;
                top: 10px;
                right: 10px;
                background-color: white;
                padding: 10px;
                border-radius: 5px;
                box-shadow: 0 0 5px rgba(0, 0, 0, 0.5);
                font-size: 14px;
            }}
        </style>
    </head>
    <body>

    <div id="map"></div>
    <div id="popup"></div>
    <div id="colorbar">
        {colorbar_html}
    </div>
    <div id="colorbar-labels">
        <span>-60 mm/年</span>
        <span>0 mm/年</span>
        <span>60 mm/年</span>
    </div>
    <div id="layerToggle">
        <label><input type="checkbox" id="insarLayerCheckbox" checked> 顯示 InSAR 圖層</label>
    </div>

    <script>
        const subscriptionKey = '3IH2ktGvtdzmhTFRQCVsOu7RsRZy1mspoK8XRWRItXLhgUNXeNfhJQQJ99AHACYeBjFAEl8IAAAgAZMPnwPm';  // Replace with your actual Azure Maps API key

        const map = new atlas.Map('map', {{
            center: [{center_lon}, {center_lat}],
            zoom: 8,
            language: 'zh-Hant',  // Set map language to Traditional Chinese
            authOptions: {{
                authType: 'subscriptionKey',
                subscriptionKey: subscriptionKey
            }},
            style: 'road'  // Using default road style
        }});

        let insarLayer; // Define a variable to hold the InSAR layer

        map.events.add('ready', function() {{
            fetch('GeoJSON_file_simplified.geojson')
                .then(response => response.json())
                .then(data => {{
                    const dataSource = new atlas.source.DataSource();
                    map.sources.add(dataSource);
                    dataSource.add(data);

                    insarLayer = new atlas.layer.PolygonLayer(dataSource, 'insar-velocity-layer', {{
                        fillColor: ['get', 'color'],
                        fillOpacity: 0.5,
                        strokeColor: 'black',
                        strokeWidth: 1
                    }});

                    map.layers.add(insarLayer);

                    map.events.add('mousemove', function(e) {{
                        const shapes = map.layers.getRenderedShapes(e.position, 'insar-velocity-layer');
                        if (shapes.length > 0) {{
                            const properties = shapes[0].getProperties();
                            const popup = document.getElementById('popup');
                            popup.style.display = 'block';
                            popup.style.left = e.position[0] + 'px';
                            popup.style.top = e.position[1] + 'px';
                            popup.innerHTML = `速度 (mm/年): ${{properties.velocity}}`;
                        }} else {{
                            document.getElementById('popup').style.display = 'none';
                        }}
                    }});

                    
                    const marker1 = new atlas.HtmlMarker({{
                        position: [{marker_lon}, {marker_lat}],  // Coordinates for Yunlin High-Speed Rail Station
                        popup: new atlas.Popup({{
                            content: '<div style="padding:5px;">{marker1_name}</div>'
                        }}),
                        icon: 'pin-round-blue'  // Use a supported icon type (e.g., 'pin-round-blue')
                    }});

                    
                    const marker2 = new atlas.HtmlMarker({{
                        position: [{marker2_lon}, {marker2_lat}],  // Coordinates for Chiayi High-Speed Rail Station
                        popup: new atlas.Popup({{
                            content: '<div style="padding:5px;">{marker2_name}</div>'
                        }}),
                        icon: 'pin-round-blue'  // Use a supported icon type (e.g., 'pin-round-blue')
                    }});

                    map.markers.add(marker1);
                    map.markers.add(marker2);

                    // Automatically open the marker popups
                    marker1.togglePopup();
                    marker2.togglePopup();
                }})
                .catch(error => console.error('Error loading GeoJSON:', error));
        }});

        // Toggle the InSAR layer visibility
        document.getElementById('insarLayerCheckbox').addEventListener('change', function(e) {{
            if (e.target.checked) {{
                map.layers.add(insarLayer);
            }} else {{
                map.layers.remove(insarLayer);
            }}
        }});
    </script>

    </body>
    </html>
    '''

    # Write the HTML content to a file
    with open('index.html', 'w') as f:
        f.write(html_content)
    print("HTML file generated. Please open 'index.html' to view the map.")

if __name__ == '__main__':
    main()





