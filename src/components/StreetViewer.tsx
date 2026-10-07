import { useEffect, useRef, useState } from 'react'
import { Viewer, SpotTag, PointGeometry, type TagComponent, type ViewerImageEvent } from 'mapillary-js'
import 'mapillary-js/dist/mapillary.css'
import { ArrowUpRight, Camera, CircleHelp, Expand, LoaderCircle, Sparkles, Users } from 'lucide-react'
import { Button } from './ui/button'
import type { PanoramaAnalysis, PanoramaDetection, Panorama } from '../types'
import { inPune } from './MapPanel'

interface Props {
  selected: Panorama | null
  analysis: PanoramaAnalysis | null
  activeDetection: PanoramaDetection | null
  scanning: boolean
  unavailable: string
  onImage: (p: Panorama) => void
  onDetection: (d: PanoramaDetection) => void
}

export function StreetViewer(props: Props) {
  const container = useRef<HTMLDivElement>(null), viewer = useRef<Viewer | null>(null), currentId = useRef(''), desiredId = useRef('')
  const latest = useRef(props); latest.current = props
  const [error, setError] = useState(''), [loading, setLoading] = useState(false), [mounted, setMounted] = useState(false), [loadedImage, setLoadedImage] = useState('')
  const [viewerRetry, setViewerRetry] = useState(0)
  const token = import.meta.env.VITE_MAPILLARY_CLIENT_TOKEN as string | undefined
  const hasSelection = Boolean(props.selected)

  useEffect(() => {
    if (!token || !container.current || !hasSelection) return
    let instance: Viewer
    let disposed = false
    setError(''); setLoadedImage(''); setLoading(true)
    try {
      instance = new Viewer({ accessToken: token, container: container.current, component: { cover: false, tag: true, sequence: true, direction: true } })
    } catch { setLoading(false); setError('Your browser could not start the 360° viewer. Try Chrome or Edge with graphics acceleration enabled.'); return }
    viewer.current = instance
    const imageChanged = ({ image }: ViewerImageEvent) => {
      if (disposed || !latest.current.selected || (desiredId.current && desiredId.current !== image.id)) return
      if (latest.current.selected.id !== currentId.current && image.id !== latest.current.selected.id) return
      const position = image.originalLngLat
      currentId.current = image.id
      setLoadedImage(image.id); setLoading(false)
      const supported = image.cameraType === 'spherical' && inPune(position.lat, position.lng)
      setError(supported ? '' : 'This image is outside the supported Pune panoramas. Choose another map point.')
      latest.current.onImage({ id: image.id, lat: position.lat, lng: position.lng, captured_at: image.capturedAt, creator: image.creatorUsername || 'Mapillary contributor', is_pano: image.cameraType === 'spherical' })
    }
    instance.on('image', imageChanged)
    instance.setFilter(['==', 'cameraType', 'spherical']).catch(() => { if (!disposed) setError('Could not apply the panorama filter. Please reopen the viewer.') })
    const resize = new ResizeObserver(() => instance.resize())
    resize.observe(container.current)
    setMounted(true)
    return () => {
      disposed = true; resize.disconnect(); instance.off('image', imageChanged); instance.remove()
      viewer.current = null; currentId.current = ''; desiredId.current = ''; setMounted(false); setLoadedImage(''); setLoading(false); setError('')
    }
  }, [token, hasSelection, viewerRetry])

  useEffect(() => {
    const id = props.selected?.id, instance = viewer.current
    if (!mounted || !id || !instance || id === currentId.current) return
    let cancelled = false
    desiredId.current = id
    setLoadedImage(''); setLoading(true); setError('')
    instance.getComponent<TagComponent>('tag').removeAll()
    instance.moveTo(id).then(() => {
      if (!cancelled) { desiredId.current = ''; setLoading(false) }
    }).catch(() => {
      if (!cancelled) { setLoading(false); setError('Mapillary could not open this photo. Check that the browser client token can access it, then retry or choose another photograph.') }
    })
    return () => { cancelled = true }
  }, [props.selected?.id, mounted, viewerRetry])

  useEffect(() => {
    const instance = viewer.current
    if (!instance) return
    const component = instance.getComponent<TagComponent>('tag')
    component.removeAll()
    const result = props.analysis
    if (!result || result.image_id !== loadedImage || result.image_id !== props.selected?.id) return
    const team = result.source === 'team'
    const tags = result.detections.map((d, i) => {
      const active = props.activeDetection?.id === d.id
      // Human labels get their own colour and wording so they are never mistaken for AI output.
      const tag = new SpotTag(d.id, new PointGeometry(d.tag), { color: active ? 0xffd166 : team ? 0x9fd3ff : 0xd8f3a5, textColor: 0x173e32, text: team ? (active ? `Team label ${i + 1}` : `${i + 1}`) : `Litter ${i + 1} · ${Math.round(d.confidence * 100)}%`, editable: false })
      tag.on('click', () => latest.current.onDetection(d))
      return tag
    })
    component.add(tags)
    return () => { if (viewer.current === instance) component.removeAll() }
  }, [props.analysis, props.selected?.id, props.activeDetection?.id, loadedImage])

  useEffect(() => {
    if (!props.activeDetection || props.analysis?.image_id !== loadedImage || props.selected?.id !== loadedImage) return
    viewer.current?.setCenter(props.activeDetection.tag)
  }, [props.activeDetection, props.analysis?.image_id, props.selected?.id, loadedImage])

  return <section className="panel street-panel" aria-label="Street panorama viewer">
    <div className="panel-heading"><span className="eyebrow"><span className="step-number">02</span> LOOK AROUND</span></div>
    <div className={`street-stage ${props.selected && token ? 'has-image' : ''}`}>
      <div ref={container} className="street-container" data-testid="street-viewer"/>
      {(!props.selected || !token) && <div className="street-empty">
        <Camera size={30} strokeWidth={1.4}/>
        <h2>No street view open</h2>
        <p>{!token ? 'Street imagery is not connected yet. You can still choose a place on the map.' : 'Pick a place or a team-labelled street, then drag to look around.'}</p>
      </div>}
      {loading && props.selected && <div className="viewer-loading" role="status"><LoaderCircle className="animate-spin" size={20}/> Opening panorama…</div>}
      {error && props.selected && <div className="viewer-error" role="alert"><CircleHelp size={18}/><span>{error}</span><Button size="sm" variant="outline" onClick={() => setViewerRetry(value => value + 1)}>Retry viewer</Button></div>}
      {props.selected && token && <Button variant="outline" size="icon" className="expand-view" aria-label="Expand street view" onClick={() => container.current?.parentElement?.requestFullscreen().catch(() => setError('Fullscreen is unavailable in this browser.'))}><Expand/></Button>}
    </div>
    <div className="scan-toolbar" role="status">
      <div><span className="scan-toolbar-title">{props.scanning ? 'Finding litter…' : props.analysis?.source === 'team' ? `${props.analysis.detections.length} litter ${props.analysis.detections.length === 1 ? 'item' : 'items'} labelled by our team` : props.analysis ? `${props.analysis.detections.length} suspected litter ${props.analysis.detections.length === 1 ? 'item' : 'items'} marked` : 'Automatic litter markings'}</span><span className="scan-toolbar-sub">{props.scanning ? 'Look around while all directions are checked.' : props.analysis?.source === 'team' ? 'Human labels, not AI. Drag to look around; select a marker to inspect it.' : props.analysis ? 'Select a marker to inspect it. Check the photo date below.' : props.unavailable || 'Open a street view to start. No scan button needed.'}</span></div>
      {props.scanning ? <LoaderCircle className="animate-spin" size={20}/> : props.analysis?.source === 'team' ? <Users size={20}/> : <Sparkles size={20}/>}
    </div>
    {props.selected && <div className="image-credit"><span>Captured {props.selected.captured_at ? new Date(props.selected.captured_at).toLocaleDateString('en-IN', { day: 'numeric', month: 'short', year: 'numeric' }) : 'date unavailable'} · {props.selected.creator}</span><a href={`https://www.mapillary.com/app/?pKey=${props.selected.id}`} target="_blank" rel="noreferrer">Mapillary · CC BY-SA <ArrowUpRight size={12}/></a></div>}
  </section>
}
