import type { WsMessage } from '@/types'

type Handler<T = unknown> = (payload: T) => void

// ── 调试开关 ────────────────────────────────────────────────────
const DEBUG_WEBSOCKET = false

/** WebSocket 长连接封装（仿真实时推送） */
export class WsClient {
  private ws: WebSocket | null = null
  private handlers = new Map<string, Handler[]>()
  private reconnectTimer: ReturnType<typeof setTimeout> | null = null

  constructor(private url: string) {}

  connect() {
    this.ws = new WebSocket(this.url)
    
    this.ws.onopen = () => {
      if (DEBUG_WEBSOCKET) {
        console.log(`[WsClient] ✅ WebSocket 已连接: ${this.url}`)
      }
    }

    this.ws.onmessage = (ev) => {
      try {
        const msg = JSON.parse(ev.data) as WsMessage
        const list = this.handlers.get(msg.type) ?? []
        list.forEach(h => h(msg.payload))
      } catch {
        console.warn('[WsClient] 无法解析消息:', ev.data)
      }
    }
    
    this.ws.onerror = (ev) => {
      console.error('[WsClient] ❌ WebSocket 错误:', ev)
    }

    this.ws.onclose = () => {
      if (DEBUG_WEBSOCKET) {
        console.warn(`[WsClient] ⚠️ WebSocket 已断开: ${this.url}`)
      }
      this.reconnectTimer = setTimeout(() => {
        if (DEBUG_WEBSOCKET) {
          console.log('[WsClient] 尝试重新连接...')
        }
        this.connect()
      }, 3000)
    }
  }

  on<T>(type: string, handler: Handler<T>) {
    const list = (this.handlers.get(type) ?? []) as Handler[]
    list.push(handler as Handler)
    this.handlers.set(type, list)
  }

  send(type: string, payload: unknown) {
    if (this.ws?.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify({ type, payload }))
    }
  }

  disconnect() {
    if (this.reconnectTimer) clearTimeout(this.reconnectTimer)
    this.ws?.close()
    this.ws = null
  }
}
