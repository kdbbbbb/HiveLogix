UAV 禁飞区 SUMO 真实路网包
===========================
高度阈值: 75.0 m
选区范围: (121.458850, 31.245920) → (121.532637, 31.308983)
路网来源: OpenStreetMap (Overpass API)
节点数:   5728
边数:     13433
坐标系:   EPSG:32651 (UTM Zone 51N)

使用方法:
  sumo-gui -n roads.net.xml --additional-files no_fly_zones.add.xml

文件说明:
  roads.net.xml         — 基于 OSM 真实道路的 SUMO 路网
  no_fly_zones.add.xml  — 建筑多边形禁飞区叠加层
  area.osm              — 原始 OSM 数据

在 sumo-gui 中，红色多边形 = 禁飞区，绿色 = 可飞区。
