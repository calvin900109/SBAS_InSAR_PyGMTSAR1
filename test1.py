# -*- coding: utf-8 -*-
import platform, sys, os

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

"""## Define Sentinel-1 SLC Scenes and Processing Parameters"""

SCENES = [
    'S1A_IW_SLC__1SDV_20230108T100127_20230108T100155_046691_0598BF_3F66',
    'S1A_IW_SLC__1SDV_20230225T100126_20230225T100154_047391_05B059_D93D',
    'S1A_IW_SLC__1SDV_20230414T100127_20230414T100155_048091_05C7F9_AD41',
    'S1A_IW_SLC__1SDV_20230601T100129_20230601T100157_048791_05DE1B_F69C',
    'S1A_IW_SLC__1SDV_20230719T100132_20230719T100200_049491_05F383_1D33',
    'S1A_IW_SLC__1SDV_20230905T100135_20230905T100203_050191_060A81_C1ED',
    'S1A_IW_SLC__1SDV_20231023T100135_20231023T100203_050891_062262_5EDD',
    'S1A_IW_SLC__1SDV_20231210T100134_20231210T100202_051591_063A8B_F40D',
    'S1A_IW_SLC__1SDV_20231222T100133_20231222T100201_051766_0640A4_0BB8'
]
SUBSWATH = 1
POLARIZATION = 'VV'

REFERENCE    = '2023-07-19'
WORKDIR      = 'raw1'
DATADIR      = 'data1'
BASEDAYS     = 100
BASEMETERS   = 150

# define DEM filename inside data directory
DEM = f'{DATADIR}/dem.nc'
LANDMASK = f'{DATADIR}/landmask.nc'
def main():
    """## Download and Unpack Datasets
    
    ## Enter Your ASF User and Password
    
    If the data directory is empty or doesn't exist, you'll need to download Sentinel-1 scenes from the Alaska Satellite Facility (ASF) datastore. Use your Earthdata Login credentials. If you don't have an Earthdata Login, you can create one at https://urs.earthdata.nasa.gov//users/new
    
    You can also use pre-existing SLC scenes stored on your Google Drive, or you can copy them using a direct public link from iCloud Drive.
    
    The credentials below are available at the time the notebook is validated.
    """
    
    # Set these variables to None and you will be prompted to enter your username and password below.
    asf_username = 'GoogleColab2023'
    asf_password = 'GoogleColab_2023'
    
    # Set these variables to None and you will be prompted to enter your username and password below.
    asf = ASF(asf_username, asf_password)
    # Optimized scene downloading from ASF - only the required subswaths and polarizations.
    print(asf.download_scenes(DATADIR, SCENES, SUBSWATH))
    
    # scan the data directory for SLC scenes and download missed orbits
    S1.download_orbits(DATADIR, S1.scan_slc(DATADIR))

    geojson = '''
    {
      "type": "Feature",
      "properties": {},
      "geometry": {
        "type": "Polygon",
        "coordinates": [
          [
            [120.0018, 23.7512],
            [120.1356, 23.1695],
            [120.8809, 23.2923],
            [120.7739, 23.8874],
            [120.0018, 23.7512]
          ]
        ]
      }
    }
    '''
    AOI = gpd.GeoDataFrame.from_features([json.loads(geojson)])
#    BUFFER = 0.025
#    # geometry is too small for the processing, enlarge it
#    AOI['geometry'] = AOI.buffer(BUFFER)
#    AOI = S1.scan_slc(DATADIR) # 使用SUBSWATH的整幅
    # download Copernicus Global DEM 1 arc-second
    Tiles().download_dem(AOI, filename=DEM)
    Tiles().download_landmask(AOI, filename=LANDMASK).plot.imshow(cmap='binary')
    """## Run Local Dask Cluster
    
    Launch Dask cluster for local and distributed multicore computing. That's possible to process terabyte scale Sentinel-1 SLC datasets on Apple Air 16 GB RAM.
    """
    
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
    
    marker_lon = 120.416389
    marker_lat = 23.736389
    ax.plot(marker_lon, marker_lat, 'ro', markersize=8)  # 'ro' 為紅色標記
    
    # 設置標題
    ax.set_title('Velocity, mm/year', fontsize=16)
    plt.suptitle('SBAS LOS Velocity, 2023', fontsize=18)
    # 自動調整佈局
    plt.tight_layout()
    # 保存圖像
    plt.savefig('SBAS LOS Velocity, 2023.jpg')

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
    with open('Yunlin_2023.geojson', 'w') as f:
        json.dump(geojson_dict, f, indent=2)
    # load the GeoJSON from the file
    with open('Yunlin_2023.geojson', 'r') as f:
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
    with open('Yunlin_2023.geojson', 'w') as f:
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
    with open('Yunlin_2023_simplified.geojson', 'w') as f:
        json.dump(geojson, f)

    # Generate the color bar matching the InSAR colormap
    colorbar_html = ''
    num_colors = 256  # Use 256 colors
    for i in range(num_colors):
        color = mcolors.to_hex(colormap(i / (num_colors - 1)))
        colorbar_html += f'<div style="flex: 1; background-color: {color};"></div>'

    # Generate the HTML file
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>InSAR Velocity Map with OpenStreetMap</title>
        <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.3/dist/leaflet.css" />
        <script src="https://unpkg.com/leaflet@1.9.3/dist/leaflet.js"></script>
        <style>
            body {{ margin: 0; padding: 0; }}
            #map {{ width: 100%; height: 100vh; position: relative; z-index: 1; }}
            #popup {{ background-color: white; border: 1px solid black; padding: 5px; position: absolute; display: none; }}
            #colorbar {{
                position: absolute;
                bottom: 20px;
                left: 20px;
                width: 300px;
                height: 20px;
                display: flex;
                z-index: 1000;  /* 確保 colorbar 顯示在地圖之上 */
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
                z-index: 1001;  /* 確保標籤顯示在地圖之上 */
            }}
        </style>
    </head>
    <body>

    <div id="map"></div>
    <div id="popup"></div>
    <div id="colorbar">
        <!-- 色條將被插入此處 -->
        {colorbar_html}
    </div>
    <div id="colorbar-labels">
        <span>-60 mm/年</span>
        <span>0 mm/年</span>
        <span>60 mm/年</span>
    </div>

    <script>
        // 初始化地圖
        const map = L.map('map').setView([23.736389, 120.416389], 8);

        // 定義不同的地圖圖層
        const osmLayer = L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
            attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
        }});

        const satelliteLayer = L.tileLayer('https://{{s}}.google.com/vt/lyrs=s&x={{x}}&y={{y}}&z={{z}}', {{
            maxZoom: 20,
            subdomains: ['mt0', 'mt1', 'mt2', 'mt3'],
            attribution: '© Google Satellite'
        }});

        // 預設使用 OSM 圖層
        osmLayer.addTo(map);

        // 載入 GeoJSON 資料
        fetch('Yunlin_2023_simplified.geojson')
            .then(response => response.json())
            .then(data => {{
                // 定義 InSAR 速度圖層樣式
                function style(feature) {{
                    return {{
                        fillColor: feature.properties.color,
                        weight: 0.5,  // 去掉黑色邊框
                        opacity: 1,
                        color: 'none',  // 不需要邊界顏色
                        fillOpacity: 0.7
                    }};
                }}

                // 將 InSAR 速度圖層加入地圖
                const insarLayer = L.geoJson(data, {{ style: style }}).addTo(map);

                // 在懸停時顯示 popup
                insarLayer.on('mouseover', function(e) {{
                    const properties = e.layer.feature.properties;
                    const roundedVelocity = properties.velocity.toFixed(2);  // 將速度縮減至小數點後兩位
                    const popup = L.popup()
                        .setLatLng(e.latlng)
                        .setContent('速度 (mm/年): ' + roundedVelocity)
                        .openOn(map);
                }});

                // 定義圖層控制器
                const baseMaps = {{
                    "OpenStreetMap": osmLayer,
                    "衛星地圖": satelliteLayer
                }};

                const overlayMaps = {{
                    "InSAR 速度圖層": insarLayer
                }};

                // 加入圖層控制器
                L.control.layers(baseMaps, overlayMaps).addTo(map);

                // 添加地標標記
                const marker1 = L.marker([23.736389, 120.416389]).addTo(map)
                    .bindPopup('雲林高鐵站').openPopup();

                const marker2 = L.marker([23.459585, 120.323124]).addTo(map)
                    .bindPopup('嘉義高鐵站').openPopup();
            }})
            .catch(error => console.error('Error loading GeoJSON:', error));
    </script>

    </body>
    </html>
    """

    # Write the HTML content to a file
    with open('index.html', 'w') as f:
        f.write(html_content)
    print("HTML file generated. Please open 'index.html' to view the map.")

if __name__ == '__main__':
    main()





