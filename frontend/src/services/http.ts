/** HTTP 请求封装 — 统一 baseURL 与错误处理 */

const BASE = import.meta.env.VITE_API_BASE ?? ''

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    headers: { 'Content-Type': 'application/json', ...(init?.headers ?? {}) },
    ...init,
  })
  if (!res.ok) {
    const err = await res.json().catch(() => ({ error: res.statusText }))
    throw new Error((err as { error: string }).error ?? res.statusText)
  }
  return res.json() as Promise<T>
}

export const http = {
  get:  <T>(path: string)                         => request<T>(path),
  post: <T>(path: string, body: unknown)          => request<T>(path, { method: 'POST', body: JSON.stringify(body) }),
}
