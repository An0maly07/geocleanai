import { ArrowUpRight, Check, CircleHelp, LoaderCircle, Sparkles, Users } from 'lucide-react'
import { Button } from './ui/button'
import type { Panorama, PanoramaAnalysis, PanoramaDetection } from '../types'

interface Props {
  selected: Panorama | null
  analysis: PanoramaAnalysis | null
  scanning: boolean
  error: string
  unavailable: string
  activeDetection: PanoramaDetection | null
  onDetection: (d: PanoramaDetection) => void
  onRetry: () => void
  teamAvailable: boolean
  modelAvailable: boolean
  showingTeam: boolean
  onMode: (mode: 'ai' | 'team') => void
}

export function ResultsPanel({ selected, analysis, scanning, error, unavailable, activeDetection, onDetection, onRetry, teamAvailable, modelAvailable, showingTeam, onMode }: Props) {
  const previewId = activeDetection?.view_id ?? analysis?.detections[0]?.view_id ?? analysis?.views[0]?.id
  const preview = analysis?.views.find(view => view.id === previewId)
  const visibleDetections = analysis?.detections.filter(d => d.view_id === previewId) ?? []
  const team = analysis?.source === 'team'
  const status = scanning ? 'FINDING LITTER' : error ? 'DETECTION FAILED' : analysis ? (team ? 'HUMAN LABELS' : 'ANALYSIS COMPLETE') : unavailable ? 'UNAVAILABLE' : selected ? 'WAITING' : 'OPEN A STREET VIEW'

  return <section className="results-panel" aria-label={team ? 'Team labels' : 'AI results'} aria-live="polite">
    <div className="results-heading"><div><span className="step-number">03</span><h2>{team ? 'Team labels' : 'Litter suggestions'}</h2></div><span className={`result-status ${analysis ? 'complete' : ''} ${team ? 'team' : ''}`}>{status}</span></div>
    {teamAvailable && <div className="mode-switch" role="group" aria-label="Choose what to mark">
      <button aria-pressed={!showingTeam} disabled={!modelAvailable} onClick={() => onMode('ai')} title={modelAvailable ? undefined : 'The trained model is not installed yet'}><Sparkles size={13}/> AI detections</button>
      <button aria-pressed={showingTeam} onClick={() => onMode('team')}><Users size={13}/> Team labels</button>
    </div>}
    {error ? <div className="scan-error"><CircleHelp size={18}/><p>{error}</p><Button size="sm" variant="outline" onClick={onRetry} disabled={Boolean(unavailable)}>Retry detection</Button></div>
      : scanning ? <div className="result-empty"><div className="scan-orb"><LoaderCircle className="animate-spin" size={23}/></div><div><strong>Finding litter…</strong><p>Checking all directions in this panorama. You can keep looking around.</p></div></div>
      : analysis ? <div className="analysis-content">
        <div className="analysis-summary"><strong>{analysis.detections.length}</strong><div>{team
          ? <><span>{analysis.detections.length === 1 ? 'litter item labelled by our team' : 'litter items labelled by our team'}</span><small>Hand-drawn by {analysis.reviewers?.join(', ') || 'the team'} · All six directions reviewed · Not AI output</small></>
          : <><span>{analysis.detections.length === 1 ? 'suspected litter item' : 'suspected litter items'}</span><small>{analysis.cached ? 'Saved result' : 'New analysis'} · {(analysis.elapsed_ms / 1000).toFixed(1)}s · All six directions checked</small></>}</div>{team ? <Users size={20}/> : <Check size={20}/>}</div>
        {!analysis.detections.length && <p className="small-note">{team ? 'The team found no visible litter in this panorama.' : 'No litter detected in this panorama. Small or hidden objects may still be present; this does not confirm the street is clean.'}</p>}
        <div className="detection-list">{analysis.detections.map((d, i) => <button key={d.id} aria-pressed={activeDetection?.id === d.id} aria-label={`Show ${team ? 'labelled' : 'suspected'} litter ${i + 1}`} className={activeDetection?.id === d.id ? 'detection active' : 'detection'} onClick={() => onDetection(d)}><span>{String(i + 1).padStart(2, '0')}</span><strong>{team ? 'Labelled litter' : 'Suspected litter'}</strong><small>{team ? `${d.view_id} view` : `${Math.round(d.confidence * 100)}% confidence`}</small><ArrowUpRight size={14}/></button>)}</div>
        {preview && <><div className="scan-preview" data-view-id={preview.id}>
          <img src={preview.preview} alt={team ? 'Photograph with the team’s litter labels' : 'Analysed photograph supporting the litter suggestions'}/>
          {visibleDetections.map(d => <span key={d.id} data-detection-id={d.id} className={`detection-box ${activeDetection?.id === d.id ? 'selected' : ''}`} style={{ left: `${d.box[0] * 100}%`, top: `${d.box[1] * 100}%`, width: `${(d.box[2] - d.box[0]) * 100}%`, height: `${(d.box[3] - d.box[1]) * 100}%` }}/>)}</div>
          <p className="small-note">{team ? 'One of six labelled directions. Select a label to look toward it.' : 'One of six analysed directions. Select a suggestion to look toward it and see its evidence.'}</p></>}
        <p className="small-note">{team ? 'Our team drew these labels by hand to train and check the model. Photos show conditions on their capture date.' : 'AI suggestions need human review. Confidence is not a measured accuracy score. Photos show conditions on their capture date.'}</p>
      </div>
      : <div className="result-empty"><div className="result-icon"><Sparkles size={21}/></div><div><strong>{unavailable ? 'Litter markings unavailable' : 'Litter is marked automatically.'}</strong><p>{unavailable || 'Choose a place and open a street view. Suspected litter will appear here and in the panorama.'}</p></div></div>}
  </section>
}
