import type { Analysis, Health, Panorama, PanoramaAnalysis, Place, TeamPanorama, ViewSpec } from '../types'
export class ApiError extends Error {
  constructor(message: string, public code: string, public status: number) { super(message) }
}
async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`/api${path}`, init)
  const body = await response.json().catch(() => null)
  if (!response.ok) throw new ApiError(body?.error?.message ?? 'The service is unavailable. Please try again.', body?.error?.code ?? 'unavailable', response.status)
  return body as T
}
export const api = {
  health: () => request<Health>('/health'),
  // Static file from scripts/build_team_labels.py; absent in builds without it.
  teamLabels: async (): Promise<Record<string, TeamPanorama>> => {
    const response = await fetch('/team-labels/index.json')
    return response.ok && response.headers.get('content-type')?.includes('json') ? response.json() : {}
  },
  search: (q: string, signal?: AbortSignal) => request<{ places: Place[] }>(`/search?q=${encodeURIComponent(q)}`, { signal }),
  panoramas: (lat: number, lng: number, signal?: AbortSignal) => request<{ panoramas: Panorama[]; truncated: boolean }>(`/panoramas?lat=${lat}&lng=${lng}`, { signal }),
  panorama: (id: string, signal?: AbortSignal) => request<Panorama>(`/panoramas/${id}`, { signal }),
  analyze: (imageId: string, view: ViewSpec, signal?: AbortSignal) => request<Analysis>('/analyze', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ image_id: imageId, view }), signal }),
  analyzePanorama: async (imageId: string, signal: AbortSignal) => {
    try {
      return await request<PanoramaAnalysis>('/analyze-panorama', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ image_id: imageId }), signal: AbortSignal.any([signal, AbortSignal.timeout(60000)]) })
    } catch (error) {
      if (!signal.aborted && error instanceof DOMException && error.name === 'TimeoutError') throw new ApiError('Litter detection took too long. Please retry.', 'analysis_timeout', 504)
      throw error
    }
  },
}
