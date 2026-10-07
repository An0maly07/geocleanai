export type Point = [number, number]
export type Vec3 = [number, number, number]
export interface Place { id: string; name: string; lat: number; lng: number }
export interface Panorama { id: string; lat: number; lng: number; captured_at: number; creator: string; distance_m?: number; is_pano: boolean; thumbnail_url?: string }
export interface ViewSpec { origin: Vec3; right: Vec3; down: Vec3; aspect: number }
export interface Detection { id: string; label: string; confidence: number; box: [number, number, number, number]; tag: Point }
export interface Analysis { image_id: string; model_version: string; detections: Detection[]; elapsed_ms: number; cached: boolean; view: ViewSpec; preview: string; captured_at: number }
export interface PanoramaDetection extends Detection { view_id: string }
export interface AnalysedView { id: string; view: ViewSpec; preview: string }
// source 'team' marks hand-drawn labels shown in place of model output; they must never be presented as AI results.
export interface PanoramaAnalysis { image_id: string; model_version: string; analysis_version: string; detections: PanoramaDetection[]; views: AnalysedView[]; elapsed_ms: number; cached: boolean; captured_at: number; source?: 'model' | 'team'; reviewers?: string[] }
export interface TeamPanorama { panorama: Panorama; analysis_version: string; detections: PanoramaDetection[]; views: AnalysedView[]; reviewers: string[]; unclear_views: string[] }
export interface Health { status: 'ready' | 'setup_required'; checks: Record<string, boolean>; model_version: string | null; samples: Panorama[] }
