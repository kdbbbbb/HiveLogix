/**
 * Vite 开发环境 Mock 中间件
 * 拦截后端API请求，返回模拟数据，使前端可以脱离后端独立运行
 */
import type { Connect } from 'vite'
import type { IncomingMessage, ServerResponse } from 'http'

// 返回统一格式的 JSON 响应
function jsonResponse(res: ServerResponse, data: object, status = 200) {
    const body = JSON.stringify(data)
    res.writeHead(status, {
        'Content-Type': 'application/json',
        'Content-Length': Buffer.byteLength(body),
        'Access-Control-Allow-Origin': '*',
        'Access-Control-Allow-Headers': 'Content-Type, batoken, ba-user-token, think-lang, server',
        'Access-Control-Allow-Methods': 'GET, POST, PUT, DELETE, OPTIONS',
    })
    res.end(body)
}

// 读取请求体
function readBody(req: IncomingMessage): Promise<string> {
    return new Promise((resolve) => {
        let data = ''
        req.on('data', (chunk) => (data += chunk))
        req.on('end', () => resolve(data))
    })
}

// ---- Mock 菜单数据 ----
const mockMenus = [
    {
        id: 1,
        name: 'dashboard',
        title: '数据看板',
        icon: 'fa fa-dashboard',
        type: 'menu',
        menu_type: 'tab',
        path: 'dashboard',
        component: '/src/views/backend/dashboard.vue',
        keepalive: 0,
        extend: '',
        children: [],
    },
    {
        id: 2,
        name: 'project',
        title: '项目管理',
        icon: 'fa fa-folder-open',
        type: 'menu',
        menu_type: 'tab',
        path: 'project',
        component: '/src/views/backend/project/index.vue',
        keepalive: 0,
        extend: '',
        children: [],
    },
    {
        id: 3,
        name: 'equipmentGroup',
        title: '设备管理',
        icon: 'fa fa-server',
        type: 'menu_dir',
        menu_type: 'tab',
        path: 'equipment',
        component: '',
        keepalive: 0,
        extend: '',
        children: [
            {
                id: 31,
                name: 'equipmentDevice',
                title: '机场管理',
                icon: 'fa fa-building',
                type: 'menu',
                menu_type: 'tab',
                path: 'equipment/device',
                component: '/src/views/backend/equipment/device/index.vue',
                keepalive: 0,
                extend: '',
                children: [],
            },
            {
                id: 32,
                name: 'equipmentAircraft',
                title: '飞行器管理',
                icon: 'fa fa-plane',
                type: 'menu',
                menu_type: 'tab',
                path: 'equipment/aircraft',
                component: '/src/views/backend/equipment/aircraft/index.vue',
                keepalive: 0,
                extend: '',
                children: [],
            },
            {
                id: 33,
                name: 'equipmentLoad',
                title: '负载管理',
                icon: 'fa fa-cubes',
                type: 'menu',
                menu_type: 'tab',
                path: 'equipment/load',
                component: '/src/views/backend/equipment/load/index.vue',
                keepalive: 0,
                extend: '',
                children: [],
            },
        ],
    },
    {
        id: 4,
        name: 'airline',
        title: '航线管理',
        icon: 'fa fa-map-o',
        type: 'menu',
        menu_type: 'tab',
        path: 'airline',
        component: '/src/views/backend/airline/index.vue',
        keepalive: 0,
        extend: '',
        children: [],
    },
    {
        id: 5,
        name: 'flighttask',
        title: '任务管理',
        icon: 'fa fa-tasks',
        type: 'menu',
        menu_type: 'tab',
        path: 'flighttask',
        component: '/src/views/backend/flighttask/index.vue',
        keepalive: 0,
        extend: '',
        children: [],
    },
    {
        id: 6,
        name: 'flightSpace',
        title: '飞行空间',
        icon: 'fa fa-globe',
        type: 'menu',
        menu_type: 'tab',
        path: 'flightSpace',
        component: '/src/views/backend/flightSpace/index.vue',
        keepalive: 0,
        extend: '',
        children: [],
    },
    {
        id: 7,
        name: 'flightrecord',
        title: '飞行记录',
        icon: 'fa fa-history',
        type: 'menu',
        menu_type: 'tab',
        path: 'flightrecord',
        component: '/src/views/backend/flightrecord/index.vue',
        keepalive: 0,
        extend: '',
        children: [],
    },
    {
        id: 8,
        name: 'logisticsMap',
        title: '物流配送地图',
        icon: 'fa fa-map-marker',
        type: 'menu',
        menu_type: 'tab',
        path: 'logisticsMap',
        component: '/src/views/backend/logisticsMap/index.vue',
        keepalive: 0,
        extend: '',
        children: [],
    },
]

// ---- 管理员信息 ----
const mockAdminInfo = {
    id: 1,
    username: 'admin',
    nickname: '超级管理员',
    avatar: '',
    last_login_time: '2026-03-16 00:00:00',
    super: true,
}

// ---- 站点配置 ----
const mockSiteConfig = {
    siteName: 'NEXUS HIVE',
    version: '1.0.0',
    cdnUrl: '',
    apiUrl: '',
    upload: { mode: 'local' },
    headNav: [],
    recordNumber: '',
    cdnUrlParams: '',
    initialize: false,
    userInitialize: false,
}

/**
 * 创建 Mock 中间件
 */
export function createMockMiddleware(): Connect.HandleFunction {
    return async (req: IncomingMessage, res: ServerResponse, next: Connect.NextFunction) => {
        const url = req.url || ''
        const method = (req.method || 'GET').toUpperCase()

        // 处理 OPTIONS 预检请求
        if (method === 'OPTIONS') {
            res.writeHead(204, {
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Headers': 'Content-Type, batoken, ba-user-token, think-lang, server',
                'Access-Control-Allow-Methods': 'GET, POST, PUT, DELETE, OPTIONS',
            })
            res.end()
            return
        }

        // ---- GET /admin/Index/login - 获取验证码状态 ----
        if (url.startsWith('/admin/Index/login') && method === 'GET') {
            console.log('[Mock] GET /admin/Index/login - 跳过验证码')
            return jsonResponse(res, { code: 1, msg: 'ok', data: { captcha: false } })
        }

        // ---- POST /admin/Index/login - 登录 ----
        if (url.startsWith('/admin/Index/login') && method === 'POST') {
            console.log('[Mock] POST /admin/Index/login - 模拟登录成功')
            return jsonResponse(res, {
                code: 1,
                msg: '登录成功',
                data: {
                    userInfo: {
                        ...mockAdminInfo,
                        token: 'mock-token-nexus-hive',
                        refresh_token: 'mock-refresh-nexus-hive',
                    },
                },
            })
        }

        // ---- GET /admin/Index/index - 后台初始化 ----
        if (url.startsWith('/admin/Index/index') && method === 'GET') {
            console.log('[Mock] GET /admin/Index/index - 返回菜单和配置')
            return jsonResponse(res, {
                code: 1,
                msg: 'ok',
                data: {
                    siteConfig: mockSiteConfig,
                    terminal: {
                        npmPackageManager: 'pnpm',
                        phpDevelopmentServer: 'builtin',
                    },
                    adminInfo: mockAdminInfo,
                    menus: mockMenus,
                },
            })
        }

        // ---- POST /admin/Index/logout - 退出登录 ----
        if (url.startsWith('/admin/Index/logout') && method === 'POST') {
            console.log('[Mock] POST /admin/Index/logout - 模拟退出')
            return jsonResponse(res, { code: 1, msg: '退出成功', data: {} })
        }

        // ---- GET /api/index/index - 前台初始化 ----
        if (url.startsWith('/api/index/index') && method === 'GET') {
            console.log('[Mock] GET /api/index/index - 前台初始化')
            return jsonResponse(res, {
                code: 1,
                msg: 'ok',
                data: {
                    site: mockSiteConfig,
                    rules: [],
                    menus: [],
                    openMemberCenter: false,
                    userInfo: null,
                },
            })
        }

        // ---- POST /admin/Index/token/refresh - Token 刷新 ----
        if (url.includes('token/refresh') && method === 'POST') {
            console.log('[Mock] Token 刷新')
            return jsonResponse(res, {
                code: 1,
                msg: 'ok',
                data: {
                    type: 'admin-refresh',
                    token: 'mock-token-nexus-hive',
                },
            })
        }

        // ---- 其他 /admin/ 或 /api/ 接口 - 返回空数据 ----
        if (url.startsWith('/admin/') || url.startsWith('/api/')) {
            console.log(`[Mock] ${method} ${url} - 返回空数据`)
            return jsonResponse(res, { code: 1, msg: 'ok', data: { list: [], total: 0, remark: '' } })
        }

        // 其他请求交给 Vite 处理
        next()
    }
}
