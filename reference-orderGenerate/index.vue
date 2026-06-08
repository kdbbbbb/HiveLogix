<template>
    <div class="default-main ba-table-box">
        <el-card class="order-generator-card" shadow="never">
            <template #header>
                <div class="generator-header">
                    <div class="title">动态订单随机生成模块</div>
                    <div class="status">
                        <el-tag :type="generatorRunning ? 'success' : 'info'">{{ generatorRunning ? '生成中' : '已停止' }}</el-tag>
                        <span class="counter">累计订单: {{ generatedOrders.length }}</span>
                    </div>
                </div>
            </template>

            <el-form label-width="130px" class="generator-form">
                <el-row :gutter="12">
                    <el-col :span="6">
                        <el-form-item label="订单生成率(单/分钟)">
                            <el-input-number v-model="generatorConfig.arrivalRate" :min="1" :max="120" :step="1" style="width: 100%" />
                        </el-form-item>
                    </el-col>
                    <el-col :span="6">
                        <el-form-item label="重量范围(kg)">
                            <div class="range-inputs">
                                <el-input-number v-model="generatorConfig.weightMin" :min="0.1" :step="0.1" style="width: 48%" />
                                <span>-</span>
                                <el-input-number v-model="generatorConfig.weightMax" :min="0.1" :step="0.1" style="width: 48%" />
                            </div>
                        </el-form-item>
                    </el-col>
                    <el-col :span="6">
                        <el-form-item label="距离范围(km)">
                            <div class="range-inputs">
                                <el-input-number v-model="generatorConfig.distanceMin" :min="0.1" :step="0.1" style="width: 48%" />
                                <span>-</span>
                                <el-input-number v-model="generatorConfig.distanceMax" :min="0.1" :step="0.1" style="width: 48%" />
                            </div>
                        </el-form-item>
                    </el-col>
                    <el-col :span="6">
                        <el-form-item label="时间窗(分钟)">
                            <div class="range-inputs">
                                <el-input-number v-model="generatorConfig.windowMin" :min="10" :step="5" style="width: 48%" />
                                <span>-</span>
                                <el-input-number v-model="generatorConfig.windowMax" :min="10" :step="5" style="width: 48%" />
                            </div>
                        </el-form-item>
                    </el-col>
                </el-row>

                <el-row :gutter="12">
                    <el-col :span="8">
                        <el-form-item label="优先级概率-紧急(%)">
                            <el-input-number v-model="generatorConfig.priorityUrgent" :min="0" :max="100" :step="1" style="width: 100%" />
                        </el-form-item>
                    </el-col>
                    <el-col :span="8">
                        <el-form-item label="优先级概率-普通(%)">
                            <el-input-number v-model="generatorConfig.priorityNormal" :min="0" :max="100" :step="1" style="width: 100%" />
                        </el-form-item>
                    </el-col>
                    <el-col :span="8">
                        <el-form-item label="优先级概率-低(%)">
                            <el-input-number v-model="generatorConfig.priorityLow" :min="0" :max="100" :step="1" style="width: 100%" />
                        </el-form-item>
                    </el-col>
                </el-row>
            </el-form>

            <div class="generator-actions">
                <el-button type="primary" @click="startGenerator" :disabled="generatorRunning">开始生成</el-button>
                <el-button @click="stopGenerator" :disabled="!generatorRunning">停止生成</el-button>
                <el-button @click="generateOnce">生成一单</el-button>
                <el-button @click="clearGeneratedOrders">清空订单</el-button>
            </div>

            <el-table :data="generatedOrders" stripe border max-height="360" style="width: 100%" class="generated-order-table">
                <el-table-column label="订单号" prop="orderId" min-width="165" />
                <el-table-column label="订单类型" prop="orderType" min-width="110" />
                <el-table-column label="优先级" min-width="120">
                    <template #default="scope">
                        <el-tag :type="priorityTagType(scope.row.priorityLevel)">{{ scope.row.priorityLabel }}</el-tag>
                    </template>
                </el-table-column>
                <el-table-column label="发布时间" prop="releaseTimeText" min-width="165" />
                <el-table-column label="期望送达" prop="expectedDeliveryTimeText" min-width="165" />
                <el-table-column label="最晚送达" prop="deadlineText" min-width="165" />
                <el-table-column label="配送时间窗" prop="timeWindowText" min-width="220" />
                <el-table-column label="仓库位置" min-width="160">
                    <template #default="scope">
                        <span>{{ scope.row.warehouseName }}</span>
                    </template>
                </el-table-column>
                <el-table-column label="目的地坐标" prop="deliveryLocationText" min-width="200" />
                <el-table-column label="重量(kg)" prop="payloadWeight" min-width="100" />
                <el-table-column label="体积(m3)" prop="payloadVolume" min-width="100" />
                <el-table-column label="价值(元)" prop="orderValue" min-width="110" />
                <el-table-column label="任务描述" prop="taskDescription" min-width="260" show-overflow-tooltip />
            </el-table>
        </el-card>

        <el-alert class="ba-table-alert" v-if="baTable.table.remark" :title="baTable.table.remark" type="info" show-icon />

        <!-- 表格顶部菜单 -->
        <!-- 自定义按钮请使用插槽，甚至公共搜索也可以使用具名插槽渲染，参见文档 -->
        <TableHeader
            :buttons="['refresh', 'add', 'edit', 'delete', 'comSearch', 'quickSearch', 'columnDisplay']"
            :quick-search-placeholder="t('Quick search placeholder', { fields: t('flighttask.quick Search Fields') })"
        ></TableHeader>

        <!-- 表格 -->
        <!-- 表格列有多种自定义渲染方式，比如自定义组件、具名插槽等，参见文档 -->
        <!-- 要使用 el-table 组件原有的属性，直接加在 Table 标签上即可 -->
        <Table ref="tableRef"></Table>

        <!-- 表单 -->
        <PopupForm />
    </div>
</template>

<script setup lang="ts">
import { ElMessage } from 'element-plus'
import { onBeforeUnmount, onMounted, provide, reactive, ref, useTemplateRef, watch } from 'vue'
import { useI18n } from 'vue-i18n'
import PopupForm from './popupForm.vue'
import { baTableApi } from '/@/api/common'
import { defaultOptButtons } from '/@/components/table'
import TableHeader from '/@/components/table/header/index.vue'
import Table from '/@/components/table/index.vue'
import baTableClass from '/@/utils/baTable'

defineOptions({
    name: 'flighttask',
})

const { t } = useI18n()
const tableRef = useTemplateRef('tableRef')
const optButtons: OptButton[] = defaultOptButtons(['edit', 'delete'])
const orderStorageKey = 'flighttask-random-orders'

interface WarehousePoint {
    id: string
    name: string
    lng: number
    lat: number
}

interface GeneratedOrder {
    orderId: string
    orderType: '普通配送' | '医疗急送' | '生鲜配送'
    priorityLevel: 1 | 2 | 3
    priorityLabel: string
    releaseTime: Date
    expectedDeliveryTime: Date
    deadline: Date
    timeWindowStart: Date
    timeWindowEnd: Date
    releaseTimeText: string
    expectedDeliveryTimeText: string
    deadlineText: string
    timeWindowText: string
    warehouseName: string
    warehouseLocation: [number, number]
    deliveryLocation: [number, number]
    deliveryLocationText: string
    payloadWeight: number
    payloadVolume: number
    orderValue: number
    taskDescription: string
}

const warehousePool: WarehousePoint[] = [
    { id: 'WH-01', name: '中心仓', lng: 104.0728, lat: 30.6658 },
    { id: 'WH-02', name: '东区仓', lng: 104.1068, lat: 30.6512 },
    { id: 'WH-03', name: '北区仓', lng: 104.0642, lat: 30.6935 },
]

const generatorConfig = reactive({
    arrivalRate: 5,
    weightMin: 0.5,
    weightMax: 5,
    distanceMin: 1,
    distanceMax: 10,
    windowMin: 60,
    windowMax: 120,
    priorityUrgent: 20,
    priorityNormal: 60,
    priorityLow: 20,
})

const generatedOrders = ref<GeneratedOrder[]>([])
const generatorRunning = ref(false)
let generatorTimer: number | null = null

const randomInRange = (min: number, max: number, precision = 2) => {
    const value = min + Math.random() * (max - min)
    return Number(value.toFixed(precision))
}

const pad2 = (value: number) => String(value).padStart(2, '0')

const formatDateTime = (date: Date) => {
    return `${date.getFullYear()}-${pad2(date.getMonth() + 1)}-${pad2(date.getDate())} ${pad2(date.getHours())}:${pad2(date.getMinutes())}:${pad2(date.getSeconds())}`
}

const clampAndFixConfig = () => {
    if (generatorConfig.weightMin > generatorConfig.weightMax) {
        const tmp = generatorConfig.weightMin
        generatorConfig.weightMin = generatorConfig.weightMax
        generatorConfig.weightMax = tmp
    }
    if (generatorConfig.distanceMin > generatorConfig.distanceMax) {
        const tmp = generatorConfig.distanceMin
        generatorConfig.distanceMin = generatorConfig.distanceMax
        generatorConfig.distanceMax = tmp
    }
    if (generatorConfig.windowMin > generatorConfig.windowMax) {
        const tmp = generatorConfig.windowMin
        generatorConfig.windowMin = generatorConfig.windowMax
        generatorConfig.windowMax = tmp
    }
}

const pickPriority = (): 1 | 2 | 3 => {
    const urgent = Math.max(generatorConfig.priorityUrgent, 0)
    const normal = Math.max(generatorConfig.priorityNormal, 0)
    const low = Math.max(generatorConfig.priorityLow, 0)
    const total = urgent + normal + low
    if (total <= 0) return 2

    const p = Math.random() * total
    if (p < urgent) return 1
    if (p < urgent + normal) return 2
    return 3
}

const priorityLabelMap: Record<1 | 2 | 3, string> = {
    1: '1-紧急',
    2: '2-普通',
    3: '3-低优先级',
}

const priorityTagType = (priorityLevel: 1 | 2 | 3) => {
    if (priorityLevel === 1) return 'danger'
    if (priorityLevel === 2) return 'warning'
    return 'info'
}

const pickOrderType = (priority: 1 | 2 | 3): '普通配送' | '医疗急送' | '生鲜配送' => {
    if (priority === 1) {
        return Math.random() > 0.35 ? '医疗急送' : '普通配送'
    }
    if (priority === 3) {
        return Math.random() > 0.5 ? '普通配送' : '生鲜配送'
    }
    return ['普通配送', '生鲜配送', '医疗急送'][Math.floor(Math.random() * 3)] as '普通配送' | '医疗急送' | '生鲜配送'
}

const calcDestinationByDistance = (warehouse: WarehousePoint, distanceKm: number): [number, number] => {
    const angle = Math.random() * Math.PI * 2
    const deltaLat = (distanceKm / 111) * Math.sin(angle)
    const deltaLng = (distanceKm / (111 * Math.cos((warehouse.lat * Math.PI) / 180))) * Math.cos(angle)
    return [Number((warehouse.lng + deltaLng).toFixed(6)), Number((warehouse.lat + deltaLat).toFixed(6))]
}

const buildOrderId = () => {
    const date = new Date()
    const stamp = `${date.getFullYear()}${pad2(date.getMonth() + 1)}${pad2(date.getDate())}${pad2(date.getHours())}${pad2(date.getMinutes())}${pad2(date.getSeconds())}`
    const randomCode = Math.floor(1000 + Math.random() * 9000)
    return `ORD-${stamp}-${randomCode}`
}

const persistGeneratedOrders = () => {
    const plainData = generatedOrders.value.map((item) => ({
        ...item,
        releaseTime: item.releaseTime.toISOString(),
        expectedDeliveryTime: item.expectedDeliveryTime.toISOString(),
        deadline: item.deadline.toISOString(),
        timeWindowStart: item.timeWindowStart.toISOString(),
        timeWindowEnd: item.timeWindowEnd.toISOString(),
    }))
    localStorage.setItem(orderStorageKey, JSON.stringify(plainData))
}

const restoreGeneratedOrders = () => {
    try {
        const raw = localStorage.getItem(orderStorageKey)
        if (!raw) return
        const parsed = JSON.parse(raw)
        if (!Array.isArray(parsed)) return

        generatedOrders.value = parsed.map((item: any) => {
            return {
                ...item,
                releaseTime: new Date(item.releaseTime),
                expectedDeliveryTime: new Date(item.expectedDeliveryTime),
                deadline: new Date(item.deadline),
                timeWindowStart: new Date(item.timeWindowStart),
                timeWindowEnd: new Date(item.timeWindowEnd),
            } as GeneratedOrder
        })
    } catch {
        generatedOrders.value = []
    }
}

const generateOrder = () => {
    clampAndFixConfig()
    const releaseTime = new Date()
    const windowMinutes = randomInRange(generatorConfig.windowMin, generatorConfig.windowMax, 0)
    const windowStartOffset = randomInRange(5, 30, 0)
    const windowStart = new Date(releaseTime.getTime() + windowStartOffset * 60 * 1000)
    const windowEnd = new Date(windowStart.getTime() + windowMinutes * 60 * 1000)
    const expectedOffset = randomInRange(3, Math.max(windowMinutes - 5, 6), 0)
    const expectedDeliveryTime = new Date(windowStart.getTime() + expectedOffset * 60 * 1000)
    const deadline = new Date(windowEnd.getTime() + randomInRange(10, 30, 0) * 60 * 1000)

    const priority = pickPriority()
    const orderType = pickOrderType(priority)
    const warehouse = warehousePool[Math.floor(Math.random() * warehousePool.length)]
    const distance = randomInRange(generatorConfig.distanceMin, generatorConfig.distanceMax, 2)
    const deliveryLocation = calcDestinationByDistance(warehouse, distance)
    const payloadWeight = randomInRange(generatorConfig.weightMin, generatorConfig.weightMax, 2)
    const payloadVolume = randomInRange(payloadWeight * 0.002, payloadWeight * 0.006, 3)
    const orderValue = randomInRange(payloadWeight * 180, payloadWeight * 460, 2)
    const orderId = buildOrderId()

    const order: GeneratedOrder = {
        orderId,
        orderType,
        priorityLevel: priority,
        priorityLabel: priorityLabelMap[priority],
        releaseTime,
        expectedDeliveryTime,
        deadline,
        timeWindowStart: windowStart,
        timeWindowEnd: windowEnd,
        releaseTimeText: formatDateTime(releaseTime),
        expectedDeliveryTimeText: formatDateTime(expectedDeliveryTime),
        deadlineText: formatDateTime(deadline),
        timeWindowText: `[${formatDateTime(windowStart)} , ${formatDateTime(windowEnd)}]`,
        warehouseName: warehouse.name,
        warehouseLocation: [warehouse.lng, warehouse.lat],
        deliveryLocation,
        deliveryLocationText: `${deliveryLocation[0]}, ${deliveryLocation[1]}`,
        payloadWeight,
        payloadVolume,
        orderValue,
        taskDescription: `${orderType} | 从${warehouse.name}发往目的地，距离约${distance}km，载荷${payloadWeight}kg，优先级${priorityLabelMap[priority]}`,
    }

    generatedOrders.value.unshift(order)
    if (generatedOrders.value.length > 300) {
        generatedOrders.value = generatedOrders.value.slice(0, 300)
    }
    persistGeneratedOrders()
}

const stopGenerator = () => {
    if (generatorTimer !== null) {
        window.clearInterval(generatorTimer)
        generatorTimer = null
    }
    generatorRunning.value = false
}

const startGenerator = () => {
    clampAndFixConfig()
    if (generatorConfig.arrivalRate <= 0) {
        ElMessage.warning('订单生成率必须大于 0')
        return
    }
    stopGenerator()

    const interval = Math.max(200, Math.floor(60000 / generatorConfig.arrivalRate))
    generatorTimer = window.setInterval(() => {
        generateOrder()
    }, interval)
    generatorRunning.value = true
    ElMessage.success(`订单生成器已启动，当前速率 ${generatorConfig.arrivalRate} 单/分钟`)
}

const generateOnce = () => {
    generateOrder()
    ElMessage.success('已生成 1 条随机订单')
}

const clearGeneratedOrders = () => {
    generatedOrders.value = []
    persistGeneratedOrders()
    ElMessage.success('订单已清空')
}

watch(
    () => generatorConfig.arrivalRate,
    () => {
        if (generatorRunning.value) {
            startGenerator()
        }
    }
)

/**
 * baTable 内包含了表格的所有数据且数据具备响应性，然后通过 provide 注入给了后代组件
 */
const baTable = new baTableClass(
    new baTableApi('/admin/Flighttask/'),
    {
        pk: 'id',
        column: [
            { type: 'selection', align: 'center', operator: false },
            { label: t('flighttask.id'), prop: 'id', align: 'center', width: 70, operator: 'RANGE', sortable: 'custom' },
            { label: t('flighttask.bid'), prop: 'bid', align: 'center', operatorPlaceholder: t('Fuzzy query'), operator: 'LIKE', sortable: false },
            { label: t('flighttask.tid'), prop: 'tid', align: 'center', operatorPlaceholder: t('Fuzzy query'), operator: 'LIKE', sortable: false },
            {
                label: t('flighttask.airline__name'),
                prop: 'airline.name',
                align: 'center',
                operatorPlaceholder: t('Fuzzy query'),
                render: 'tags',
                operator: 'LIKE',
            },
            {
                label: t('flighttask.equipment__nickname'),
                prop: 'equipment.nickname',
                align: 'center',
                operatorPlaceholder: t('Fuzzy query'),
                render: 'tags',
                operator: 'LIKE',
            },
            {
                label: t('flighttask.execute_time'),
                prop: 'execute_time',
                align: 'center',
                render: 'datetime',
                operator: 'RANGE',
                sortable: 'custom',
                width: 160,
                timeFormat: 'yyyy-mm-dd hh:MM:ss',
            },
            {
                label: t('flighttask.task_type'),
                prop: 'task_type',
                align: 'center',
                render: 'tag',
                operator: 'eq',
                sortable: false,
                replaceValue: { '0': t('flighttask.task_type 0'), '1': t('flighttask.task_type 1'), '2': t('flighttask.task_type 2') },
            },
            {
                label: t('flighttask.file_fingerprint'),
                prop: 'file_fingerprint',
                align: 'center',
                operatorPlaceholder: t('Fuzzy query'),
                operator: 'LIKE',
                sortable: false,
            },
            { label: t('flighttask.rth_altitude'), prop: 'rth_altitude', align: 'center', operator: 'RANGE', sortable: false },
            {
                label: t('flighttask.rth_mode'),
                prop: 'rth_mode',
                align: 'center',
                render: 'tag',
                operator: 'eq',
                sortable: false,
                replaceValue: { '0': t('flighttask.rth_mode 0'), '1': t('flighttask.rth_mode 1') },
            },
            {
                label: t('flighttask.out_of_control_action'),
                prop: 'out_of_control_action',
                align: 'center',
                render: 'tag',
                operator: 'eq',
                sortable: false,
                replaceValue: {
                    '0': t('flighttask.out_of_control_action 0'),
                    '1': t('flighttask.out_of_control_action 1'),
                    '2': t('flighttask.out_of_control_action 2'),
                },
            },
            {
                label: t('flighttask.exit_wayline_when_rc_lost'),
                prop: 'exit_wayline_when_rc_lost',
                align: 'center',
                render: 'tag',
                operator: 'eq',
                sortable: false,
                replaceValue: {
                    '0': t('flighttask.exit_wayline_when_rc_lost 0'),
                    '1': t('flighttask.exit_wayline_when_rc_lost 1'),
                    '2': t('flighttask.exit_wayline_when_rc_lost 2'),
                },
            },
            {
                label: t('flighttask.wayline_precision_type'),
                prop: 'wayline_precision_type',
                align: 'center',
                render: 'tag',
                operator: 'eq',
                sortable: false,
                replaceValue: { '0': t('flighttask.wayline_precision_type 0'), '1': t('flighttask.wayline_precision_type 1') },
            },
            {
                label: t('flighttask.create_time'),
                prop: 'create_time',
                align: 'center',
                render: 'datetime',
                operator: 'RANGE',
                sortable: 'custom',
                width: 160,
                timeFormat: 'yyyy-mm-dd hh:MM:ss',
            },
            {
                label: t('flighttask.update_time'),
                prop: 'update_time',
                align: 'center',
                render: 'datetime',
                operator: 'RANGE',
                sortable: 'custom',
                width: 160,
                timeFormat: 'yyyy-mm-dd hh:MM:ss',
            },
            { label: t('Operate'), align: 'center', width: 100, render: 'buttons', buttons: optButtons, operator: false },
        ],
        dblClickNotEditColumn: [undefined],
    },
    {
        defaultItems: { task_type: '0', rth_mode: '0', out_of_control_action: '0', exit_wayline_when_rc_lost: '0', wayline_precision_type: '0' },
    }
)

provide('baTable', baTable)

onMounted(() => {
    restoreGeneratedOrders()

    baTable.table.ref = tableRef.value
    baTable.mount()
    baTable.getData()?.then(() => {
        baTable.initSort()
        baTable.dragSort()
    })
})

onBeforeUnmount(() => {
    stopGenerator()
})
</script>

<style scoped lang="scss">
.order-generator-card {
    margin-bottom: 14px;
}

.generator-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 12px;

    .title {
        font-size: 16px;
        font-weight: 600;
    }

    .status {
        display: flex;
        align-items: center;
        gap: 10px;

        .counter {
            font-size: 13px;
            color: var(--el-text-color-secondary);
        }
    }
}

.generator-form {
    margin-bottom: 8px;
}

.range-inputs {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 6px;
    width: 100%;
}

.generator-actions {
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 10px;
}

.generated-order-table {
    :deep(.el-table__cell) {
        padding: 6px 0;
    }
}

@media screen and (max-width: 1200px) {
    .generator-actions {
        flex-wrap: wrap;
    }
}
</style>
