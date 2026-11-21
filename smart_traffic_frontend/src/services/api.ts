import type { DashboardSummary, LiveTraffic, Vehicle, Violation } from "../types";
const API_BASE = (
  import.meta.env.VITE_API_BASE_URL || "http://127.0.0.1:8000"
).replace(/\/$/, "");export const CAMERA_STREAM_URL = import.meta.env.VITE_CAMERA_STREAM_URL || `${API_BASE}/api/camera/stream`;
async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, { ...init, headers: { "Content-Type": "application/json", ...(init?.headers || {}) } });
  if (!response.ok) { const message = await response.text().catch(() => ""); throw new Error(`${response.status} ${response.statusText}${message ? ` — ${message.slice(0, 180)}` : ""}`); }
  return response.json() as Promise<T>;
}
export const api = {
  health: () => request<{ status?: string; message?: string }>("/api/health"),
  live: () => request<LiveTraffic>("/api/traffic/live"),
  vehicles: () => request<Vehicle[]>("/api/traffic/vehicles"),
  density: () => request<{ density?: string | number; [key: string]: unknown }>("/api/traffic/density"),
  events: () => request<Violation[]>("/api/traffic/events"),
  violations: (params = "") => request<Violation[]>(`/api/violations${params}`),
  dashboard: () => request<DashboardSummary>("/api/dashboard/summary"),
  chat: (message: string, conversationId?: string) => request<{ answer?: string; response?: string; message?: string; conversation_id?: string; tool_calls?: string[] }>("/api/agent/chat", { method: "POST", body: JSON.stringify({ message, conversation_id: conversationId }) })
};