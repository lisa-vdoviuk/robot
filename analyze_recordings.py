"""Offline HEAR analysis. Usage: python analyze_hear.py archive.zip --out figures"""
import argparse
import csv
import io
import json
import wave
import zipfile
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

KINDS = ['quiet', 'motors', 'speaker', 'voice', 'knock', 'background']
N = 8192
HOP = 2048
BANDS = [(20,80),(80,250),(250,2000),(2000,8000),(8000,24000)]


def db(power):
    return 10*np.log10(np.maximum(power, 1e-16))


def stft(x, fs, n=N, hop=HOP):
    frames = np.lib.stride_tricks.sliding_window_view(x, n)[::hop]
    frames = frames - frames.mean(axis=1, keepdims=True)
    window = np.hanning(n)
    p = abs(np.fft.rfft(frames*window, axis=1))**2
    p /= fs*np.sum(window**2)
    p[:, 1:-1] *= 2
    return np.fft.rfftfreq(n, 1/fs), (np.arange(len(p))*hop+n/2)/fs, p


def save(fig, out, name):
    fig.savefig(out/f'{name}.png', dpi=300, bbox_inches='tight')
    fig.savefig(out/f'{name}.pdf', bbox_inches='tight')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('archive', type=Path)
    parser.add_argument('--out', type=Path, default=Path('hear_figures'))
    parser.add_argument('--margin-db', type=float, default=6.0)
    args = parser.parse_args()
    if not np.isfinite(args.margin_db) or args.margin_db < 0:
        parser.error('margin-db must be finite and nonnegative')
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    records = []
    with zipfile.ZipFile(args.archive) as z:
        if z.testzip() is not None:
            raise ValueError('ZIP integrity check failed')
        for name in sorted(z.namelist()):
            if not name.endswith('/report.json'):
                continue
            meta = json.loads(z.read(name))
            if meta.get('status') != 'complete':
                continue
            kind = next((k for k in KINDS if k in meta['scene'].lower()), None)
            if kind is None:
                raise ValueError(f'Unknown scene: {meta["scene"]}')
            with wave.open(io.BytesIO(z.read(name.rsplit('/',1)[0]+'/audio.wav'))) as w:
                fs = w.getframerate()
                if w.getnchannels()!=1 or w.getsampwidth()!=2 or fs!=48000:
                    raise ValueError('Expected 48 kHz mono PCM16 WAV')
                x = np.frombuffer(w.readframes(w.getnframes()), '<i2').astype(float)/32768
            if len(x) < fs:
                raise ValueError('Need at least one second per recording')
            f,t,p = stft(x,fs)
            records.append(dict(meta=meta,kind=kind,x=x,fs=fs,f=f,t=t,p=p))
    if not records:
        raise ValueError('No completed audio recordings')
    quiet = [r for r in records if r['kind']=='quiet']
    if not quiet:
        raise ValueError('A quiet reference recording is required')
    q = np.concatenate([r['p'] for r in quiet])
    f = records[0]['f']
    df = f[1]-f[0]
    qmean = q.mean(axis=0)
    mask = (f>=80)&(f<8000)
    qlevels = db(q[:,mask].sum(axis=1)*df)
    threshold = float(np.percentile(qlevels,95)+args.margin_db)
    # Longer windows for a separate low-frequency estimate; no zero-padding.
    fine = np.mean([stft(r['x'],r['fs'],48000,24000)[2].mean(axis=0)
                    for r in quiet],axis=0)
    ff = np.fft.rfftfreq(48000,1/48000)
    low = (ff>=20)&(ff<=250)
    low_peak = float(ff[low][np.argmax(fine[low])])
    selected = {k: next(r for r in records if r['kind']==k)
                for k in KINDS if any(r['kind']==k for r in records)}
    rows = []
    for r in records:
        x,p = r['x'],r['p']
        power = p.mean(axis=0)
        r['band_levels'] = db(p[:,mask].sum(axis=1)*df)
        row = dict(id=r['meta']['id'],scene=r['meta']['scene'],kind=r['kind'],
                   selected=selected[r['kind']] is r,notes=r['meta'].get('notes',''),
                   duration_s=len(x)/r['fs'],rms_dbfs=float(db(np.mean(x*x))),
                   peak_dbfs=float(db(np.max(x*x))),
                   near_full_scale_percent=float(100*np.mean(abs(x)>=.999)),
                   dc_offset_fs=float(x.mean()),
                   band80_8000_dbfs=float(db(power[mask].sum()*df)),
                   excess_over_quiet_db=float(db(power[mask].sum())-db(qmean[mask].sum())),
                   above_threshold_frames_percent=float(100*np.mean(r['band_levels']>threshold)))
        row['crest_factor_db']=row['peak_dbfs']-row['rms_dbfs']
        for lo,hi in BANDS:
            band=(f>=lo)&(f<hi)
            row[f'band_{lo}_{hi}_dbfs']=float(db(power[band].sum()*df))
        rows.append(row)
    with (out/'metrics_all.csv').open('w',newline='',encoding='utf-8-sig') as file:
        writer=csv.DictWriter(file,fieldnames=rows[0].keys())
        writer.writeheader();writer.writerows(rows)
    summary=dict(selection_rule='First completed take per semantic scene in sorted archive path order',
                 selected={k:r['meta']['id'] for k,r in selected.items()},
                 quiet_ids=[r['meta']['id'] for r in quiet],
                 quiet_low_peak_hz=low_peak,low_peak_bin_spacing_hz=1,
                 band80_8000_quiet_dbfs=float(db(qmean[mask].sum()*df)),
                 diagnostic_threshold_dbfs=threshold,margin_db=args.margin_db,
                 n_fft=N,hop=HOP,bin_spacing_hz=float(df),metrics=rows)
    (out/'analysis.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig, axes=plt.subplots(2,3,figsize=(13,7.5),layout='constrained')
    for ax,k in zip(axes.flat,KINDS):
        if k not in selected:
            ax.set_visible(False);continue
        r=selected[k]
        im=ax.pcolormesh(r['t'],f,db(r['p']).T,shading='auto',cmap='magma',vmin=-110,vmax=-35,rasterized=True)
        ax.set_yscale('log');ax.set_ylim(20,20000)
        ax.set_xlabel('Time (s)');ax.set_ylabel('Frequency (Hz)')
        ax.set_title(k.capitalize())
    fig.colorbar(im,ax=axes.ravel().tolist(),label='PSD (dB re 1 FS²/Hz)')
    fig.suptitle('HEAR pilot | first completed take per scene | shared colour scale')
    save(fig,out,'01_spectrograms_overview')
    for k,r in selected.items():
        fig,axs=plt.subplots(3,1,figsize=(10,9),layout='constrained')
        im=axs[0].pcolormesh(r['t'],f,db(r['p']).T,shading='auto',cmap='magma',vmin=-110,vmax=-35,rasterized=True)
        axs[0].set_yscale('log');axs[0].set_ylim(20,20000)
        axs[0].set(xlabel='Time (s)',ylabel='Frequency (Hz)',title=k.capitalize()+' | spectrogram')
        fig.colorbar(im,ax=axs[0],label='PSD (dB re 1 FS²/Hz)')
        axs[1].plot(r['t'],r['band_levels'],label='80–8000 Hz window level')
        axs[1].axhline(threshold,color='crimson',ls='--',label=f'Quiet P95 + {args.margin_db:g} dB = {threshold:.1f} dBFS')
        axs[1].set(xlabel='Time (s)',ylabel='Band RMS (dBFS)',ylim=(-80,0))
        axs[1].legend(loc='upper right',fontsize=8)
        axs[2].semilogx(f[1:],db(r['p'].mean(axis=0)[1:]),label=k)
        axs[2].semilogx(f[1:],db(qmean[1:]),color='black',alpha=.7,label='Quiet reference mean')
        axs[2].set(xlim=(20,20000),ylim=(-130,-25),xlabel='Frequency (Hz)',ylabel='PSD (dB re 1 FS²/Hz)')
        axs[2].legend();axs[2].grid(alpha=.2)
        fig.suptitle(r['meta']['id'],fontsize=10)
        save(fig,out,'detail_'+k)
    fig,ax=plt.subplots(figsize=(10,4),layout='constrained')
    ax.plot(ff[low],db(fine[low]),label='Mean quiet PSD, 1 s Hann windows')
    ax.axvline(low_peak,color='crimson',ls='--',label=f'Strongest 20–250 Hz bin: {low_peak:g} Hz')
    ax.set(xlabel='Frequency (Hz)',ylabel='PSD (dB re 1 FS²/Hz)',title='Low-frequency background component')
    ax.legend();ax.grid(alpha=.2)
    save(fig,out,'02_quiet_low_frequency')
    report=f'''# HEAR pilot spectral analysis

Analysed {len(records)} complete recordings. Figures use the first completed take per semantic scene, not the best-looking result. Both quiet takes form the reference. Unique IDs distinguish duplicate scene names. See analysis.json for selected IDs and metrics_all.csv for all takes.

Strongest quiet bin between 20 and 250 Hz: {low_peak:.1f} Hz with 1 Hz bin spacing. This identifies a recorded component, not its physical source or a universal noise cutoff. The recording includes room, microphone, mounting and possible device processing effects.

Mean quiet level in 80–8000 Hz: {summary['band80_8000_quiet_dbfs']:.2f} dBFS. Exploratory activity threshold: {threshold:.2f} dBFS (95th percentile of quiet frame band levels plus {args.margin_db:g} dB). This is a level threshold within a stated band, not a frequency threshold, speech detector, or validated operating point. It is fitted to these quiet recordings; independent recordings are needed to assess false alarms. Above-threshold percentages describe overlapping analysis windows, not classification accuracy or independent events.

Method: mono PCM16 divided by 32768; no amplitude normalisation or filtering. Spectrogram: {N}-sample Hann windows, hop {HOP}, per-window mean removal, one-sided power spectral density. At 48 kHz this is {N/48000*1000:.2f} ms per window, {HOP/48000*1000:.2f} ms per hop and {48000/N:.5f} Hz bin spacing (not the full spectral resolving power). Colour scale is identical across scenes, -110 to -35 dB re 1 FS²/Hz; values outside are colour-clipped. Figures display 20 Hz to 20 kHz; band metrics extend to Nyquist. Band levels integrate PSD in linear units before conversion to dBFS. A full-scale sine has RMS approximately -3.01 dBFS in this convention.

Excess over quiet compares whole-recording 80–8000 Hz power with the pooled quiet reference. It is NOT SNR: isolated clean signal and simultaneous noise are unavailable, and matching gain/AGC/conditions is unverified. Silence within an event recording dilutes this whole-clip measure. Near-full-scale counts cannot rule out analog clipping or limiting upstream. Digital levels are not calibrated dB SPL.

Dataset notes: H06 notes mention quiet music, quiet talking and opening shelves, so it is mixed background, not a speech-free negative. Speaker notes specify 440 Hz; a spectral peak elsewhere does not invalidate that note because background and harmonics may dominate. Most notes lack distance and gain details. Do not infer microphone frequency response, causal noise sources, or human presence from these spectra alone.

Next controlled checks: repeat quiet with motors off and unchanged gain; compare microphone mounting positions; document fan and power conditions; record speech at a fixed distance. Test any high-pass filter offline on copies and compare speech and knocks before changing capture. Preserve raw WAV files.
'''
    report += '\n## All recordings\n\n| Scene | RMS dBFS | Peak dBFS | 80–8000 Hz dBFS | Excess over quiet dB |\n|---|---:|---:|---:|---:|\n'
    for row in rows:
        report += f"| {row['id']} | {row['rms_dbfs']:.2f} | {row['peak_dbfs']:.2f} | {row['band80_8000_dbfs']:.2f} | {row['excess_over_quiet_db']:.2f} |\n"
    (out/'report.md').write_text(report,encoding='utf-8')
    print(json.dumps({k:v for k,v in summary.items() if k!='metrics'},indent=2))


if __name__=='__main__':
    main()
