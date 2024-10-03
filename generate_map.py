import json
import geopandas as gpd
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt

# 讀取 GeoJSON 檔案並簡化幾何形狀
with open('Yunlin_2023.geojson', 'r') as f:
    geojson = json.load(f)

# 使用 Geopandas 簡化幾何，保留 95% 的形狀
gdf = gpd.GeoDataFrame.from_features(geojson["features"])
gdf["geometry"] = gdf["geometry"].simplify(tolerance=0.0003, preserve_topology=True)

# 更新 GeoJSON 資料
geojson = json.loads(gdf.to_json())

# 定義顏色映射（colormap）
colormap = plt.get_cmap('turbo')
norm = plt.Normalize(vmin=-60, vmax=60)  # 假設速度範圍是 -60 至 60 mm/年

def velocity_to_color(velocity, limits=[-60, 60]):
    """將速度轉換為顏色"""
    normalized = (velocity - limits[0]) / (limits[1] - limits[0])
    return mcolors.to_hex(colormap(normalized))

# 對每個 feature 應用顏色
for feature in geojson['features']:
    color = velocity_to_color(feature['properties']['velocity'])
    feature['properties']['color'] = color

# 保存簡化後的 GeoJSON 檔案
with open('Yunlin_2023_simplified.geojson', 'w') as f:
    json.dump(geojson, f)

# 生成對應 InSAR 色譜的色條
colorbar_html = ''
num_colors = 256  # 使用 256 種顏色
for i in range(num_colors):
    color = mcolors.to_hex(colormap(i / (num_colors - 1)))
    colorbar_html += f'<div style="flex: 1; background-color: {color};"></div>'

# 生成 HTML 檔案內容
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
                const popup = L.popup()
                    .setLatLng(e.latlng)
                    .setContent('速度 (mm/年): ' + properties.velocity)
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

# 將 HTML 內容寫入檔案
with open('Yunlin_2023.html', 'w') as f:
    f.write(html_content)

print("HTML file generated. Please open 'Yunlin_2023.html' to view the map.")
