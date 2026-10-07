import { lazy, Suspense, useCallback, useEffect, useRef, useState } from 'react'
import { ArrowRight, ArrowUpRight, CircleHelp, Crosshair, ExternalLink, Leaf, LoaderCircle, MapPin, Moon, Search, ShieldCheck, Sun, X } from 'lucide-react'
import { Button } from './components/ui/button'
import { InfoDialog } from './components/InfoDialog'
import { ResultsPanel } from './components/ResultsPanel'
import { api } from './lib/api'
import { setTheme, useTheme } from './lib/theme'
import type { PanoramaAnalysis, PanoramaDetection, Health, Panorama, Place, TeamPanorama } from './types'

const MapPanel = lazy(() => import('./components/MapPanel'))
const StreetViewer = lazy(() => import('./components/StreetViewer').then(m => ({ default: m.StreetViewer })))
const areas: Place[] = [
  { id: 'central', name: 'Central Pune', lat: 18.5204, lng: 73.8567 },
  { id: 'shivajinagar', name: 'Shivajinagar', lat: 18.5314, lng: 73.8446 },
  { id: 'koregaon', name: 'Koregaon Park', lat: 18.5362, lng: 73.8939 },
  { id: 'kothrud', name: 'Kothrud', lat: 18.5074, lng: 73.8077 },
]

function message(error: unknown) { return error instanceof Error ? error.message : 'Something went wrong. Please retry.' }
function isAbort(error: unknown) { return error instanceof DOMException && error.name === 'AbortError' }

export default function App() {
  const [health, setHealth] = useState<Health | null>(null), [healthError, setHealthError] = useState(false), [infoOpen, setInfoOpen] = useState(false)
  const [query, setQuery] = useState(''), [places, setPlaces] = useState<Place[]>([]), [searching, setSearching] = useState(false), [searchDone, setSearchDone] = useState(false)
  const [focus, setFocus] = useState<Place | null>(null), [panoramas, setPanoramas] = useState<Panorama[]>([]), [loadingPanos, setLoadingPanos] = useState(false), [lookedUp, setLookedUp] = useState(false), [truncated, setTruncated] = useState(false)
  const [selected, setSelected] = useState<Panorama | null>(null), [notice, setNotice] = useState('')
  const [analysis, setAnalysis] = useState<PanoramaAnalysis | null>(null), [scanning, setScanning] = useState(false), [scanError, setScanError] = useState(''), [activeDetection, setActiveDetection] = useState<PanoramaDetection | null>(null)
  const [retry, setRetry] = useState(0)
  const [team, setTeam] = useState<Record<string, TeamPanorama>>({}), [mode, setMode] = useState<'ai' | 'team'>('ai'), [allTeam, setAllTeam] = useState(false)
  const theme = useTheme()
  const viewerColumn = useRef<HTMLDivElement>(null), searchAbort = useRef<AbortController | null>(null), panoAbort = useRef<AbortController | null>(null), scanAbort = useRef<AbortController | null>(null)
  const searchSerial = useRef(0), panoSerial = useRef(0), scanSerial = useRef(0), selectedId = useRef('')

  const refreshHealth = useCallback(() => { api.health().then(h => { setHealth(h); setHealthError(false) }).catch(() => setHealthError(true)) }, [])
  useEffect(() => { api.teamLabels().then(setTeam).catch(() => setTeam({})) }, [])
  useEffect(() => { refreshHealth(); return () => { searchAbort.current?.abort(); panoAbort.current?.abort(); scanAbort.current?.abort() } }, [refreshHealth])
  const resetScan = useCallback(() => { scanSerial.current++; scanAbort.current?.abort(); setScanning(false); setAnalysis(null); setActiveDetection(null); setScanError('') }, [])
  const inspectDetection = useCallback((d: PanoramaDetection) => setActiveDetection({ ...d }), [])
  const selectPanorama = useCallback((p: Panorama) => {
    if (selectedId.current !== p.id) resetScan()
    selectedId.current = p.id; setSelected(p)
    if (window.matchMedia('(max-width: 850px)').matches) viewerColumn.current?.scrollIntoView({ behavior: window.matchMedia('(prefers-reduced-motion: reduce)').matches ? 'instant' : 'smooth', block: 'start' })
  }, [resetScan])
  const onViewerImage = useCallback((p: Panorama) => {
    if (!selectedId.current) return
    if (selectedId.current !== p.id) resetScan()
    selectedId.current = p.id; setSelected(previous => previous?.id === p.id ? { ...previous, ...p } : p)
  }, [resetScan])

  const lookup = useCallback(async (place: Place) => {
    searchSerial.current++; searchAbort.current?.abort(); setSearching(false)
    resetScan(); selectedId.current = ''; setSelected(null)
    panoAbort.current?.abort(); const controller = new AbortController(); panoAbort.current = controller
    const sequence = ++panoSerial.current
    setFocus(place); setQuery(place.name.split(',')[0]); setPlaces([]); setSearchDone(false); setLoadingPanos(true); setNotice(''); setPanoramas([]); setLookedUp(false); setTruncated(false)
    try {
      const result = await api.panoramas(place.lat, place.lng, controller.signal)
      if (sequence !== panoSerial.current) return
      setPanoramas(result.panoramas); setTruncated(result.truncated); setLookedUp(true)
    } catch (error) { if (!isAbort(error) && sequence === panoSerial.current) setNotice(message(error)) }
    finally { if (sequence === panoSerial.current) setLoadingPanos(false) }
  }, [resetScan])
  const onMapPoint = useCallback((lat: number, lng: number) => { void lookup({ id: `${lat},${lng}`, name: 'Selected map point', lat, lng }) }, [lookup])
  const submitSearch = async (event: React.FormEvent) => {
    event.preventDefault(); if (query.trim().length < 2) return
    searchAbort.current?.abort(); const controller = new AbortController(); searchAbort.current = controller
    const sequence = ++searchSerial.current
    setSearching(true); setNotice(''); setPlaces([]); setSearchDone(false)
    try {
      const result = await api.search(query.trim(), controller.signal)
      if (sequence === searchSerial.current) {
        if (result.places.length === 1) void lookup(result.places[0])
        else { setPlaces(result.places); setSearchDone(true) }
      }
    } catch (error) { if (!isAbort(error) && sequence === searchSerial.current) setNotice(message(error)) }
    finally { if (sequence === searchSerial.current) setSearching(false) }
  }
  const configured = Boolean(health?.checks.model && health?.checks.mapillary && health?.checks.shared_cache)
  const browserToken = Boolean(import.meta.env.VITE_MAPILLARY_CLIENT_TOKEN)
  const supported = Boolean(selected?.is_pano && selected.lat >= 18.40 && selected.lat <= 18.70 && selected.lng >= 73.70 && selected.lng <= 74.05)
  const teamEntry = selected ? team[selected.id] : undefined
  // Team labels show when chosen, or when no model is available for this panorama.
  const showTeam = Boolean(teamEntry && supported && browserToken && (mode === 'team' || !configured || healthError))
  const unavailable = showTeam ? '' : healthError ?'Litter detection is unavailable while the service is offline.' : !health ? 'Checking litter detection availability…' : !configured ? 'Litter markings are unavailable until the trained model and service connections are ready.' : !browserToken ? 'Street imagery is unavailable until the image connection is ready.' : selected && !supported ? 'Litter detection is available only for panoramas inside the Pune demo area.' : ''

  useEffect(() => {
    if (showTeam && selected && teamEntry) {
      resetScan()
      const labels: PanoramaAnalysis = { image_id: selected.id, model_version: 'team-labels', analysis_version: teamEntry.analysis_version, detections: teamEntry.detections, views: teamEntry.views, elapsed_ms: 0, cached: true, captured_at: teamEntry.panorama.captured_at, source: 'team', reviewers: teamEntry.reviewers }
      setAnalysis(labels); setActiveDetection(labels.detections[0] ?? null)
      return
    }
    if (!selected?.id || !supported || !configured || !browserToken || healthError) { resetScan(); return }
    const sequence = ++scanSerial.current, imageId = selected.id
    const controller = new AbortController(); scanAbort.current = controller
    setScanning(true); setScanError(''); setAnalysis(null); setActiveDetection(null)
    api.analyzePanorama(imageId, controller.signal).then(result => {
      if (controller.signal.aborted || sequence !== scanSerial.current || selectedId.current !== imageId) return
      if (result.image_id !== imageId) throw new Error('The results did not match this photograph. Please retry.')
      setAnalysis(result); setActiveDetection(result.detections[0] ?? null)
    }).catch(error => {
      if (!controller.signal.aborted && !isAbort(error) && sequence === scanSerial.current) setScanError(message(error))
    }).finally(() => { if (sequence === scanSerial.current) setScanning(false) })
    return () => { controller.abort(); scanSerial.current++ }
  }, [selected?.id, supported, configured, browserToken, healthError, health?.model_version, retry, resetScan, showTeam, teamEntry])

  const nearest = panoramas[0]
  const teamList = Object.values(team).sort((a, b) => b.detections.length - a.detections.length || a.panorama.id.localeCompare(b.panorama.id))
  const teamButton = (entry: TeamPanorama) => <Button key={entry.panorama.id} variant="outline" size="sm" className={selected?.id === entry.panorama.id ? 'active' : ''} onClick={() => { setMode('team'); selectPanorama(entry.panorama) }}>{entry.detections.length ? `${entry.detections.length} litter ${entry.detections.length === 1 ? 'label' : 'labels'}` : 'No litter labelled'} · {new Date(entry.panorama.captured_at).toLocaleDateString('en-IN', { month: 'short', year: 'numeric' })}<ArrowUpRight/></Button>

  return <div className="app-shell">
    <header className="site-header">
      <a className="brand" href="/" aria-label="GeoCleanAI home"><span className="brand-symbol"><Leaf size={23} strokeWidth={1.8} /></span><span>GeoClean<span className="brand-ai">AI</span></span></a>
      <nav aria-label="Main navigation"><button onClick={() => setInfoOpen(true)}>How it works</button>
        <button className="theme-toggle" onClick={() => setTheme(theme === 'dark' ? 'light' : 'dark')} aria-label={theme === 'dark' ? 'Switch to light mode' : 'Switch to dark mode'} title={theme === 'dark' ? 'Light mode' : 'Dark mode'}>{theme === 'dark' ? <Sun size={17}/> : <Moon size={17}/>}</button></nav>
    </header>

    <main id="explore">
      <section className="intro">
        <div><h1>A clearer view of <span>Pune.</span></h1><p>Find a place, open its street view, and look around for marked litter.</p></div>
      </section>

      <section className="explore-toolbar" aria-label="Location search">
        <div className="search-block"><form onSubmit={submitSearch} className="search-form"><Search size={19}/><input aria-label="Search Pune" placeholder="Search a street, neighbourhood or landmark…" value={query} onChange={e => { setQuery(e.target.value); setSearchDone(false); setPlaces([]); searchSerial.current++; searchAbort.current?.abort(); setSearching(false) }} maxLength={150}/><Button type="submit" disabled={searching || query.trim().length < 2} size="sm">{searching ? <LoaderCircle className="animate-spin"/> : <ArrowRight/>}<span className="sr-only">Search</span></Button></form>
          {(places.length > 0 || searchDone) && <div className="search-results" aria-live="polite">{places.length ? places.map(p => <button key={p.id} onClick={() => { setQuery(p.name.split(',')[0]); void lookup(p) }}><MapPin size={16}/><span>{p.name}</span><ArrowUpRight size={14}/></button>) : <p>No matching places in the Pune demo area. Try a nearby landmark.</p>}</div>}
        </div>
        <div className="area-shortcuts">{areas.map(area => <button key={area.id} onClick={() => { void lookup(area) }}>{area.name}<ArrowUpRight size={12}/></button>)}</div>
      </section>

      {(healthError || health?.status === 'setup_required' || !browserToken) && <div className="setup-banner" role="status"><CircleHelp size={17}/><p>{healthError ? 'The API is not responding yet. The map is available while the service starts.' : !browserToken || !health?.checks.mapillary ? 'Street imagery is not connected yet. You can still explore the Pune map.' : !health?.checks.model ? 'Street exploration is available. Automatic litter markings will appear when the trained model is ready.' : 'A service connection still needs setup before all features are available.'}</p><button onClick={() => { refreshHealth(); setInfoOpen(true) }}>View status <ArrowUpRight size={13}/></button></div>}
      {notice && <div className="notice" role="alert"><CircleHelp size={17}/><span>{notice}</span><button aria-label="Dismiss notice" onClick={() => setNotice('')}><X size={16}/></button></div>}

      <div className="workspace">
        <div className="map-column"><Suspense fallback={<div className="panel panel-loading">Loading map…</div>}><MapPanel panoramas={panoramas} selected={selected} focus={focus} onPoint={onMapPoint} onSelect={selectPanorama}/></Suspense>
          {focus && <section className="location-card" aria-label="Selected location">
            <div className="location-card-title"><MapPin size={20}/><div><span>YOUR PLACE</span><h2>{focus.name}</h2></div></div>
            <p>{loadingPanos ? 'Finding nearby street photographs…' : nearest ? `${panoramas.length} nearby street ${panoramas.length === 1 ? 'view' : 'views'}${truncated ? ' in this subset' : ''}. Nearest photo: ${nearest.distance_m ?? 0} m away.` : lookedUp ? 'No street photographs found nearby. Your place is still marked on the map.' : 'Street photo availability could not be checked.'}</p>
            {nearest ? <><Button onClick={() => selectPanorama(nearest)} disabled={!browserToken}>Open street view <ArrowUpRight size={16}/></Button>{!browserToken && <p>Street imagery is not connected yet.</p>}</> : !loadingPanos && !lookedUp && <Button variant="outline" onClick={() => void lookup(focus)}>Retry nearby photos</Button>}
          </section>}
          {(lookedUp || loadingPanos) && <section className="nearby-panel" aria-label="Available panoramas"><div className="nearby-heading"><span>Nearby street views</span><span>{loadingPanos ? 'Searching…' : lookedUp ? `${panoramas.length} available` : 'Not checked'}</span></div>
            {loadingPanos ? <div className="nearby-empty"><LoaderCircle className="animate-spin" size={18}/> Looking within 500 metres…</div> : panoramas.length ? <><div className="panorama-list">{panoramas.map(p => <button key={p.id} className={selected?.id === p.id ? 'panorama-card active' : 'panorama-card'} onClick={() => selectPanorama(p)}><div className="panorama-thumb">{p.thumbnail_url ? <img src={p.thumbnail_url} alt="Street panorama preview" loading="lazy"/> : <MapPin size={20}/>}</div><div><strong>{p.distance_m ?? 0} m away</strong><span>{p.captured_at ? new Date(p.captured_at).toLocaleDateString('en-IN', { month: 'short', year: 'numeric' }) : 'Date unavailable'}</span></div><ArrowUpRight size={16}/></button>)}</div>{truncated && <p className="small-note">A subset is shown. Select a closer map point to explore more images.</p>}</> : <div className="nearby-empty"><Crosshair size={19}/><span>{lookedUp ? 'No panoramas found within 500 m. Try another point nearby.' : 'Select a map point or an area above to check image coverage.'}</span></div>}
          </section>}
        </div>
        <div className="viewer-column" ref={viewerColumn}>
          <Suspense fallback={<div className="panel panel-loading">Loading street viewer…</div>}>
            <StreetViewer selected={selected} analysis={analysis} activeDetection={activeDetection} scanning={scanning} unavailable={unavailable} onImage={onViewerImage} onDetection={inspectDetection}/>
          </Suspense>
          <ResultsPanel selected={selected} analysis={analysis} scanning={scanning} error={scanError} unavailable={unavailable} activeDetection={activeDetection} onDetection={inspectDetection} onRetry={() => setRetry(value => value + 1)} teamAvailable={Boolean(teamEntry && supported && browserToken)} modelAvailable={configured && !healthError} showingTeam={showTeam} onMode={setMode}/>
        </div>
      </div>

      {Boolean(health?.samples.length) && <section className="verified-samples"><h2>Verified presentation locations</h2>{health!.samples.map(p => <Button key={p.id} variant="outline" onClick={() => selectPanorama(p)}>Open photo {p.id}<ArrowUpRight/></Button>)}</section>}
      {teamList.length > 0 && <section className="team-samples" aria-label="Team-labelled street views"><div className="team-heading"><h2>Team-labelled street views</h2><button onClick={() => setAllTeam(value => !value)}>{allTeam ? 'Show fewer' : `Show all ${teamList.length}`}</button></div>
        <p>Streets our team labelled by hand. Open one to see the labels{configured ? ', then switch to AI detections to compare' : ''}.</p>
        <div>{(allTeam ? teamList : teamList.slice(0, 8)).map(teamButton)}</div>
      </section>}
    </main>
    <footer><span><Leaf size={15}/> GeoCleanAI <span className="footer-dot">·</span> Built for a cleaner tomorrow.</span><div><a href="https://www.mapillary.com/" target="_blank" rel="noreferrer">Mapillary <ExternalLink size={11}/></a><a href="https://www.openstreetmap.org/copyright" target="_blank" rel="noreferrer">© OpenStreetMap contributors</a><button onClick={() => setInfoOpen(true)}><ShieldCheck size={13}/> About this prototype</button></div></footer>
    <InfoDialog open={infoOpen} onOpenChange={setInfoOpen} health={health}/>
  </div>
}

