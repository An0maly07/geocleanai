import { useEffect, useRef, useState } from 'react'
import * as maplibregl from 'maplibre-gl'
import type { Map as LibreMap, Marker } from 'maplibre-gl'
import workerUrl from 'maplibre-gl/dist/maplibre-gl-worker.mjs?worker&url'
import 'maplibre-gl/dist/maplibre-gl.css'
import { AlertCircle, Compass, LocateFixed, MapPin } from 'lucide-react'
import type { Panorama, Place } from '../types'
import { Button } from './ui/button'
import { currentTheme, useTheme, type Theme } from '../lib/theme'

export const PUNE_CENTER: [number, number] = [73.8567, 18.5204]
export const PUNE_BOUNDS: [[number, number], [number, number]] = [[73.70, 18.40], [74.05, 18.70]]
maplibregl.setWorkerUrl(workerUrl)
const mapStyle = (theme: Theme) => `https://tiles.openfreemap.org/styles/${theme === 'dark' ? 'dark' : 'liberty'}`
export function inPune(lat: number, lng: number) { return lat >= 18.40 && lat <= 18.70 && lng >= 73.70 && lng <= 74.05 }

interface Props { panoramas: Panorama[]; selected: Panorama | null; focus: Place | null; onPoint: (lat: number, lng: number) => void; onSelect: (p: Panorama) => void }
export default function MapPanel(props: Props) {
  const container = useRef<HTMLDivElement>(null), mapRef = useRef<LibreMap | null>(null), markers = useRef<Marker[]>([]), placeMarker = useRef<Marker | null>(null)
  const latest = useRef(props); latest.current = props
  const [error, setError] = useState(false), [ready, setReady] = useState(false)
  const theme = useTheme(), styled = useRef(currentTheme())
  useEffect(() => {
    if (!container.current) return
    let map: LibreMap
    try {
      map = new maplibregl.Map({ container: container.current, style: mapStyle(styled.current), center: PUNE_CENTER, zoom: 12.5, maxBounds: PUNE_BOUNDS, minZoom: 10, maxZoom: 19, attributionControl: { compact: true } })
    } catch { setError(true); return }
    mapRef.current = map
    map.addControl(new maplibregl.NavigationControl({ showCompass: false }), 'bottom-right')
    map.on('load', () => { setReady(true); setError(false) })
    map.on('error', () => setError(true))
    map.on('click', event => { if (inPune(event.lngLat.lat, event.lngLat.lng)) latest.current.onPoint(event.lngLat.lat, event.lngLat.lng) })
    const observer = new ResizeObserver(() => map.resize())
    observer.observe(container.current)
    return () => { observer.disconnect(); markers.current.forEach(m => m.remove()); placeMarker.current?.remove(); map.remove(); mapRef.current = null }
  }, [])
  useEffect(() => {
    // Pins are DOM markers, so they survive a basemap style swap.
    if (!mapRef.current || styled.current === theme) return
    styled.current = theme
    mapRef.current.setStyle(mapStyle(theme))
  }, [theme])
  useEffect(() => {
    const map = mapRef.current
    if (!map || !ready) return
    markers.current.forEach(m => m.remove())
    markers.current = props.panoramas.map(p => {
      const button = document.createElement('button')
      button.type = 'button'; button.className = `pano-marker ${props.selected?.id === p.id ? 'selected' : ''}`
      button.setAttribute('aria-label', `Open panorama ${p.id}`)
      button.title = `${p.distance_m ?? 0} m from your selection`
      button.addEventListener('click', event => { event.stopPropagation(); latest.current.onSelect(p) })
      return new maplibregl.Marker({ element: button }).setLngLat([p.lng, p.lat]).addTo(map)
    })
  }, [props.panoramas, props.selected?.id, ready])
  useEffect(() => {
    const location = props.focus
    const map = mapRef.current
    placeMarker.current?.remove(); placeMarker.current = null
    if (!location || !map || !inPune(location.lat, location.lng)) return
    const pin = document.createElement('div')
    pin.className = 'place-marker'; pin.setAttribute('role', 'img'); pin.setAttribute('aria-label', `Selected place: ${location.name}`)
    pin.title = location.name
    pin.appendChild(document.createElement('span'))
    pin.addEventListener('click', event => event.stopPropagation())
    placeMarker.current = new maplibregl.Marker({ element: pin, anchor: 'bottom' }).setLngLat([location.lng, location.lat]).addTo(map)
    map.flyTo({ center: [location.lng, location.lat], zoom: 16, duration: 900 })
    return () => { placeMarker.current?.remove(); placeMarker.current = null }
  }, [props.focus, ready])
  useEffect(() => {
    const location = props.selected
    if (location && mapRef.current && inPune(location.lat, location.lng)) mapRef.current.flyTo({ center: [location.lng, location.lat], zoom: 16, duration: 900 })
  }, [props.selected?.id, ready])
  return <section className="panel map-panel" aria-label="Pune location map">
    <div className="panel-heading"><span className="eyebrow"><span className="step-number">01</span> FIND A STREET</span></div>
    <div className="map-stage">
      <div ref={container} className="map-container" data-testid="map-container" />
      {!ready && !error && <div className="map-loading"><Compass className="animate-spin" size={20} /><span>Opening Pune map…</span></div>}
      {error && <div className="map-notice"><AlertCircle size={16} /> Map tiles are unavailable. You can still use search or the area shortcuts.</div>}
      <div className="map-top-note"><span className="live-dot" />{props.focus ? 'Your place is pinned. Open a nearby street view.' : 'Search a place or choose a point on the map'}</div>
      <Button variant="outline" size="icon" className="recenter" title="Recenter on Pune" aria-label="Recenter on Pune" onClick={() => mapRef.current?.flyTo({ center: PUNE_CENTER, zoom: 12.5 })}><LocateFixed /></Button>
      <div className="map-key"><span className="key-dot place"/> Your place <span className="key-dot" /> Street photo <span className="key-dot outlined" /> Open photo</div>
    </div>
  </section>
}
