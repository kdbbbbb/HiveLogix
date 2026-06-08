/**
 * geoLayout — 坐标布局纯函数
 *
 * 约束：
 *   - 纯函数，不 import 任何 Store / Service / Vue 响应式对象
 *   - 不持有任何状态，可直接用于单元测试
 *   - Bbox 结构与 stores/scene.ts SceneBounds 结构相同（结构兼容，无 import 依赖）
 */

/**
 * WGS84 经纬度边界框
 * 字段名与 stores/scene.ts SceneBounds 对齐，传入 SceneBounds 实例时 TypeScript
 * 结构类型检查自动兼容，无需额外转换。
 */
export interface Bbox {
  min_lng: number
  max_lng: number
  min_lat: number
  max_lat: number
}

export interface GeoLayoutOptions {
  /**
   * 随机扰动幅度，相对于格子宽 / 高的比例，取值 0~0.5
   * 默认 0.15（±15%），防止点落在格子边界上
   */
  jitter?: number
}

/**
 * 均匀栅格布局：将 n 个点均匀分散在 bbox 内，每点在格子中心附近随机扰动。
 *
 * 算法：
 *   rows = ceil(sqrt(n))
 *   cols = ceil(n / rows)
 *   将 bbox 等分为 rows × cols 格子
 *   每点 = 格子中心 + rand(±jitter × 格子宽/高)
 *   越界坐标截断到 bbox 边界内
 *
 * @param n       需要生成的点数量（≥1）
 * @param bbox    目标边界框（WGS84）
 * @param options 可选配置（jitter 扰动幅度）
 * @returns       长度恰好为 n 的坐标数组
 */
export function gridLayout(
  n: number,
  bbox: Bbox,
  options?: GeoLayoutOptions,
): Array<{ lng: number; lat: number }> {
  if (n <= 0) return []

  const jitter = options?.jitter ?? 0.15
  const rows   = Math.ceil(Math.sqrt(n))
  const cols   = Math.ceil(n / rows)
  const cellW  = (bbox.max_lng - bbox.min_lng) / cols
  const cellH  = (bbox.max_lat - bbox.min_lat) / rows

  const points: Array<{ lng: number; lat: number }> = []

  outer: for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      if (points.length >= n) break outer

      const centerLng = bbox.min_lng + (c + 0.5) * cellW
      const centerLat = bbox.min_lat + (r + 0.5) * cellH
      const dLng      = (Math.random() * 2 - 1) * jitter * cellW
      const dLat      = (Math.random() * 2 - 1) * jitter * cellH

      points.push({
        lng: Math.max(bbox.min_lng, Math.min(bbox.max_lng, centerLng + dLng)),
        lat: Math.max(bbox.min_lat, Math.min(bbox.max_lat, centerLat + dLat)),
      })
    }
  }

  return points
}

/**
 * 联合均匀栅格布局：将 nd 个仓库 + ns 个充换电站分配到同一均匀网格上。
 *
 * 约束满足：
 *   1. 仓库 + 充换电站整体形成均匀分布（所有点来自同一 N_total 网格）
 *   2. 仓库自身也形成均匀分布（通过 2D 子网格映射选取仓库格点位置）
 *
 * 算法：
 *   rows = ceil(sqrt(N_total)); cols = ceil(N_total / rows)
 *   仓库子网格 dRows×dCols，将 (dr, dc) 映射到完整网格位置
 *     tr = round(dr * (rows-1) / (dRows-1))
 *     tc = round(dc * (cols-1) / (dCols-1))
 *   仓库占据映射后的格点，充换电站填充剩余格点
 *
 * @param nd      仓库数量（≥0）
 * @param ns      充换电站数量（≥0）
 * @param bbox    目标边界框（WGS84）
 * @param options 可选配置（jitter 扰动幅度）
 */
export function gridLayoutSplit(
  nd: number,
  ns: number,
  bbox: Bbox,
  options?: GeoLayoutOptions,
): {
  depots:   Array<{ lng: number; lat: number }>
  stations: Array<{ lng: number; lat: number }>
} {
  const ntotal = nd + ns
  if (ntotal === 0) return { depots: [], stations: [] }
  if (nd === 0) return { depots: [], stations: gridLayout(ns, bbox, options) }
  if (ns === 0) return { depots: gridLayout(nd, bbox, options), stations: [] }

  const jitter = options?.jitter ?? 0.12

  // 完整网格尺寸
  const rows = Math.ceil(Math.sqrt(ntotal))
  const cols = Math.ceil(ntotal / rows)

  // 仓库子网格尺寸（用于映射到完整网格）
  const dRows = Math.ceil(Math.sqrt(nd))
  const dCols = Math.ceil(nd / dRows)

  // 2D 子网格映射：将仓库子网格格点对应到完整网格索引
  const depotSet = new Set<number>()
  let dCount = 0
  outerD: for (let dr = 0; dr < dRows; dr++) {
    for (let dc = 0; dc < dCols; dc++) {
      if (dCount >= nd) break outerD
      const tr = dRows > 1 ? Math.round(dr * (rows - 1) / (dRows - 1)) : Math.floor(rows / 2)
      const tc = dCols > 1 ? Math.round(dc * (cols - 1) / (dCols - 1)) : Math.floor(cols / 2)
      depotSet.add(tr * cols + tc)
      dCount++
    }
  }
  // 边界保护：若映射碰撞导致集合不足 nd 个，顺序补充
  for (let i = 0; depotSet.size < nd && i < rows * cols; i++) {
    depotSet.add(i)
  }

  // 生成所有格点坐标并按类型分配
  const cellW = (bbox.max_lng - bbox.min_lng) / cols
  const cellH = (bbox.max_lat - bbox.min_lat) / rows
  const depotCoords:   Array<{ lng: number; lat: number }> = []
  const stationCoords: Array<{ lng: number; lat: number }> = []

  let pos = 0
  outer: for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      if (pos >= ntotal) break outer
      const cx  = bbox.min_lng + (c + 0.5) * cellW
      const cy  = bbox.min_lat + (r + 0.5) * cellH
      const lng = Math.max(bbox.min_lng, Math.min(bbox.max_lng,
        cx + (Math.random() * 2 - 1) * jitter * cellW))
      const lat = Math.max(bbox.min_lat, Math.min(bbox.max_lat,
        cy + (Math.random() * 2 - 1) * jitter * cellH))
      if (depotSet.has(pos)) depotCoords.push({ lng, lat })
      else                   stationCoords.push({ lng, lat })
      pos++
    }
  }

  return { depots: depotCoords, stations: stationCoords }
}
