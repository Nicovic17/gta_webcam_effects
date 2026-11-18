# GTA Webcam Effects

Progetto autonomo separato da Webcam Memes. Avvio principale: **gta.py**.
Non importa file da `webcam_memes` o `disappear/local_app`.

## Avvio rapido

Su questo PC usa **AVVIA_GTA.bat**, oppure apri PowerShell nella cartella
del progetto ed esegui:

```powershell
.\.venv\Scripts\python.exe -B gta.py
```

Il menu compare prima di aprire la telecamera:

| Tasto | Modalita |
|---|---|
| `1` | WASTED Webcam: replay rallentato, scritta WASTED, audio e respawn. |
| `2` | WANTED Webcam: stelle, minimappa, scansione, foto WANTED e polizia. |
| `3` | VICE MODE AI Webcam - in progress: singolo scatto trasformato tramite OpenAI. |
| `4` | VICE Locale - GPU - in progress: video AI continuo con prompt libero. |

Puoi scegliere anche con il mouse. `Q` / `ESC` nel menu annullano l'avvio.
Per avviare direttamente una modalita: `gta.py --mode wanted`
(valori: `wasted`, `wanted`, `vice`, `vice_local`).

## Installazione

Windows, webcam e OBS Studio con OBS Virtual Camera sono necessari.
Per ricreare l'ambiente principale con `uv` e Python 3.13:

```powershell
.\SETUP_GTA.ps1
```

Oppure crea un virtualenv Python 3.13 e installa `requirements.txt` con pip.
Gli audio usano `pygame` se disponibile, altrimenti `ffplay` nel PATH.
Sul computer corrente e disponibile ffplay. Il worker di VICE Locale usa
un ambiente separato Python 3.10 in `.vice-local/`: questo runtime e i suoi
modelli scaricati sono esclusi da Git. Per ricrearlo serve
`SETUP_VICE_LOCALE.ps1`, che scarica le dipendenze e i modelli mancanti.
Non installare le dipendenze GPU nell'ambiente principale.

## Webcam virtuale e controlli

Avvia prima GTA Webcam Effects e poi seleziona **OBS Virtual Camera** in
Teams, Zoom, Discord o Meet. Non avviare contemporaneamente la Virtual Camera
dall'interfaccia OBS: un solo processo alla volta puo gestirla.

La richiesta alla webcam resta 1280x720 MJPEG; anteprima e uscita OBS sono
960x540 con limite di 30 FPS. L'acquisizione conserva solo il frame recente,
come nella versione precedente. Gli FPS effettivi dipendono da webcam ed effetto.

In tutte le modalita: `M` o il pulsante **Flip** specchiano la webcam,
`F` cambia fullscreen, `Q` / `ESC` chiudono. `SPAZIO` e `R` controllano
l'effetto come spiegato sotto. I pulsanti e i suggerimenti restano soltanto
nell'anteprima; OBS riceve il video pulito. Per cambiare modalita riavvia gta.py.

## Effetti

### WASTED Webcam

Seleziona **WASTED Webcam**, oppure premi `1` nel menu. Premi **SPAZIO** con
l'anteprima attiva: un impatto avvia un breve replay della webcam al 22% della
velocita, zoom verso il volto, inclinazione cinematografica, bianco e nero,
vignettatura e bande nere. Compare **WASTED** in rosso, con una frase ironica
diversa a ogni attivazione. Dopo circa 5,25 secondi una breve transizione
digitale **RESPAWN** riporta automaticamente al video dal vivo.

Il pulsante **IT / ENG** in alto a destra nell'anteprima, oppure il tasto `L`,
cambia la lingua delle frasi sotto **WASTED** tra italiano e inglese, anche
durante una sequenza. La lingua iniziale e l'italiano; la scritta WASTED resta uguale.

`SPAZIO` durante una sequenza non la riavvia; `R` la annulla immediatamente.
`M` cambia lo specchio (annullando l'eventuale sequenza attiva), `F` cambia
il fullscreen e `Q` / `ESC` chiudono. Funziona anche su OBS Virtual Camera.
Il flip specchia tutta la scena, comprese WASTED, le frasi e RESPAWN.
I suggerimenti e il pulsante Flip sono visibili soltanto nell'anteprima.
Se il volto non viene rilevato, lo zoom usa il centro dell'inquadratura.

Il replay conserva circa 0,8 secondi di fotogrammi ridotti esclusivamente
in RAM; non registra file e libera le immagini al reset/respawn. Il volto
viene analizzato su un worker separato, usando il modello gia incluso.
L'audio `assets/wasted.mp3` accompagna l'impatto;
se l'audio non e disponibile l'effetto visivo continua a funzionare.
La grafica e generata localmente. Moduli: `wasted_effect.py`
e `webcam_wasted.py`; i testi si personalizzano nelle liste `REASONS` (italiano)
e `REASONS_EN` (inglese), mantenendo le traduzioni nello stesso ordine.

### WANTED Webcam

Seleziona **WANTED Webcam**, oppure premi `2`. La webcam resta dal vivo con
cinque stelle bianche in alto a destra (`assets/whitestar.png`) e la mappa
in basso a sinistra (`assets/map.png`), sempre presenti.

**SPAZIO** avvia la modalita ricercato: le stelle passano istantaneamente dal
bianco al giallo usando `assets/yellowstar.png`, una ogni 1,6 secondi.
Ogni nuova stella gialla lampeggia per 2 secondi. Raggiunte le cinque stelle,
lampeggiano tutte insieme, in modo sincronizzato, per 3 secondi; poi restano
gialle fisse. Alla quinta
stella (circa 8 secondi) parte una scansione della persona di circa un secondo.
Viene acquisito un fotogramma in RAM e il volto piu recentemente rilevato
compare ritagliato in un riquadro quadrato a destra. L'intero riquadro
(foto, bordo, sfondo e scritta) esegue uno zoom-in/zoom-out continuo e
morbido, con un ciclo di circa 1,8 secondi e ampiezza dell'8%, mantenendo
fisso il proprio centro. La scritta **WANTED** lampeggia sotto la foto;
bagliori rossi e blu si muovono tra angoli e bordi della schermata.

La foto resta fissa mentre la webcam continua a muoversi. Non viene salvata
su disco. Se manca un rilevamento del volto, viene usata l'ultima posizione
nota oppure un ritaglio centrale. `assets/gtapolice.mp3` parte soltanto quando
tutte e cinque le stelle sono gialle (circa 8 secondi dopo **SPAZIO**) e viene
riprodotto in loop fino alla disattivazione della modalita, a `R` o all'uscita.
Premi di nuovo **SPAZIO**, oppure `R`, per
azzerare la ricerca, cancellare la foto e tornare alle cinque stelle bianche.
Stelle e mappa restano visibili; il pulsante Flip o il tasto `M` specchiano
tutta la scena, comprese scritte, stelle, mappa e foto WANTED, sia
nell'anteprima sia su OBS Virtual Camera. Modulo: `wanted_effect.py`.

### VICE MODE AI Webcam

Seleziona **VICE MODE AI Webcam - in progress**, oppure premi `3`. **SPAZIO** scatta una
foto alla risoluzione acquisita dalla camera e la invia direttamente all'API
immagini OpenAI. Il prompt richiede
un'illustrazione curata in stile artwork/loading screen GTA, mantenendo volto,
capelli, espressione, vestiti, posa, proporzioni e posizione nella foto.

Durante la generazione la webcam continua a funzionare; nell'anteprima viene
mostrato il tempo di attesa. Quando l'immagine arriva, una dissolvenza di circa
2 secondi passa dallo scatto originale alla versione generata. Il risultato
resta una **fotografia illustrata**, non un filtro video in tempo reale.
Compare anche su OBS Virtual Camera, senza scritte di avanzamento sul video.

`SPAZIO` mentre si attende non invia altre richieste. A risultato pronto,
`SPAZIO` scatta una nuova foto. **R** torna subito alla webcam e scarta il
risultato di eventuali richieste precedenti; `M` cambia lo specchio e azzera
la trasformazione, `F` cambia fullscreen e `Q` / `ESC` chiudono.
Annullare la visualizzazione non annulla una generazione gia ricevuta dal
server: il risultato viene ignorato e si attende la conclusione della richiesta
prima di inviarne un'altra. Ogni nuovo scatto inviato usa il tuo account API.

La chiave viene letta da **OPENAI_API_KEY** nell'ambiente. In alternativa,
copia `.env.example` in `.env` nella cartella dell'app e inseriscila li.
Se non e configurata, all'avvio della modalita compare un campo mascherato:
la chiave inserita resta in memoria per la sessione. `.env` e escluso da Git.
Non inserire la chiave nei file Python. Le impostazioni opzionali sono
`OPENAI_IMAGE_MODEL` (default `gpt-image-2`) e `OPENAI_IMAGE_QUALITY`
(default `high`; possibili `low`, `medium`, `high`, `auto`). L'output richiesto
e `1536x864`. Non vengono eseguiti tentativi automatici a pagamento.

Serve Internet e un progetto OpenAI abilitato alla generazione immagini, con
credito/limiti sufficienti. Errori di chiave, rete, permessi o quota appaiono
nell'anteprima. Modifiche a chiave, modello o qualita richiedono il riavvio
dell'app. Foto e risultato restano in RAM per la durata della richiesta e
dell'effetto, e non vengono salvati automaticamente.
La richiesta usa [l'endpoint ufficiale Images Edit](https://developers.openai.com/api/reference/resources/images/methods/edit).
Moduli: `vice_openai.py`, `vice_ai_effect.py`, `webcam_vice_ai.py`.

Entrambe le modalita usano `M` per lo specchio, `F` per il fullscreen,
`Q` / `ESC` per uscire. Tasti e pulsanti richiedono l'anteprima attiva.
Il risultato viene pubblicato anche sulla webcam virtuale; i suggerimenti
dei comandi compaiono solo nell'anteprima. WANTED analizza il volto localmente
su un worker separato. VICE AI non carica i modelli MediaPipe: genera una
singola immagine online solo dopo lo scatto richiesto dall'utente.
Moduli: `webcam_neon.py`, `gta_webcam_effects.py`, `neon_scene.py`.

### VICE Locale - GPU (sperimentale)

La selezione **4** trasforma continuamente i nuovi fotogrammi sulla GPU,
usando StreamDiffusion, SD-Turbo e Tiny Autoencoder. La webcam resta live.
Il worker AI usa un ambiente Python 3.10 separato in `.vice-local/`, senza
modificare le dipendenze delle altre modalita. Sono richiesti Windows x64,
una GPU NVIDIA CUDA con memoria sufficiente e driver compatibili. La prova
locale e stata eseguita su **RTX 4070 SUPER da 12 GB** con xformers;
TensorRT non e utilizzato.

Su questo computer il runtime e gia installato. Per ripetere l'installazione,
servono `uv`, `git`, Internet e diversi GB liberi. Apri PowerShell nella
cartella dell'app ed esegui:

```powershell
.\SETUP_VICE_LOCALE.ps1
```

Lo script installa Python e librerie solo in `.vice-local/`, scarica i
modelli pubblici da Hugging Face e misura le prestazioni con un'immagine
gia presente nel progetto. La cartella e esclusa da Git. Il setup e l'unico
passaggio che scarica modelli: durante l'uso il worker li carica dalla cache
locale in modalita offline. Non richiede una chiave OpenAI e non invia foto
o video a un servizio online.

Avvia `gta.py` come al solito, scegli **4 - VICE Locale - GPU** e attendi
il caricamento iniziale del modello. Nel frattempo l'anteprima resta live.

Si apre anche il pannello **Cosa vuoi diventare?**: scrivi nel campo di testo
lo stile, il personaggio o l'ambiente che vuoi ottenere. Il prompt iniziale
e quello GTA; il testo personalizzato lo sostituisce completamente.
Per esempio: `A cyberpunk portrait with neon lights`, `A watercolor painting`
oppure `A comic book superhero with bold black outlines`.

Il testo viene applicato automaticamente dopo circa **0,65 secondi senza
digitare**. Puoi anche premere **Invio** o **Applica** per inviarlo subito;
**Shift+Invio** inserisce una nuova riga. Usa **Avvia effetto** nel pannello
oppure SPAZIO nell'anteprima per avviare la trasformazione. Puoi continuare
a riscrivere il prompt mentre il video e attivo, senza riavviare l'app o
ricaricare il modello. Il pulsante **Ripristina GTA** ripristina la descrizione
iniziale. Il pannello include anche il selettore dell'intensita.

I prompt vuoti o oltre 1000 caratteri non vengono applicati: resta valido
l'ultimo testo accettato. Preferisci descrizioni brevi in inglese, perche
il modello ha un contesto testuale limitato. Il modello interpreta la
descrizione e la combina con la webcam; la corrispondenza esatta dipende
dall'immagine, dal testo e dall'intensita scelta. Non e un assistente di chat.
Testo e impostazioni restano solo nella sessione. Se chiudi il pannello,
il video continua: premi **P nell'anteprima** per riaprirlo.

| Tasto | Azione |
|---|---|
| `SPAZIO` | Attiva/disattiva la trasformazione continua, con ingresso graduale. |
| `1` / `2` / `3` | Intensita leggera / media / forte; inizialmente media. |
| `P` | Riapre il pannello per modificare il prompt. |
| `R` | Disattiva e scarta i risultati in corso. |
| `M` | Specchio ON/OFF. |
| `F` | Fullscreen ON/OFF. |
| `Q` / `ESC` | Esce e libera il processo GPU, webcam e OBS. |

La generazione interna e `512x320`; l'immagine viene adattata mantenendo
l'inquadratura e inviata a OBS nella risoluzione video configurata. Camera
e AI lavorano separatamente: viene conservato solo l'ultimo frame in attesa,
senza accumulare una coda di fotogrammi vecchi. I comandi e i tempi compaiono
solo nell'anteprima locale. Un errore GPU ripristina il video originale;
i dettagli tecnici sono in `.vice-local/worker.log`.

La verifica del 17 novembre 2026 ha misurato circa **17 frame AI/s** e
**73 ms di latenza mediana** nel collegamento completo con il Python
dell'app e un'immagine di test in movimento. L'inferenza GPU richiedeva circa
49 ms. Sono misure di laboratorio, non una garanzia per ogni webcam o con
altri programmi sulla GPU. L'uscita video puo aggiornarsi piu spesso dei
risultati AI disponibili.

Questa prova privilegia il movimento e il basso ritardo: il risultato puo
ammorbidire o alterare il volto e sfarfallare tra frame. La fedelta al volto
e lo stile GTA richiedono ancora affinamento; non raggiunge la qualita dello
scatto elaborato da OpenAI. Prova le tre intensita per confrontare stabilita
e trasformazione. La precedente modalita **3 - VICE MODE AI Webcam** resta
disponibile per la trasformazione di una singola foto.

Riferimenti: [StreamDiffusion](https://github.com/cumulo-autumn/StreamDiffusion),
[SD-Turbo](https://huggingface.co/stabilityai/sd-turbo),
[Tiny Autoencoder](https://huggingface.co/madebyollin/taesd).
Moduli: `vice_local_engine.py`, `vice_local_worker.py`, `vice_local_client.py`,
`vice_local_settings.py`, `vice_prompt_panel.py`, `webcam_vice_local.py`.
Il campo prompt aggiorna gli embedding testuali con `update_prompt`, come
nella [demo realtime img2img](https://github.com/cumulo-autumn/StreamDiffusion/tree/main/demo/realtime-img2img),
sul worker GPU separato; i risultati del prompt precedente vengono scartati.
I controlli del pannello non compaiono nel video OBS.
Dipendenze isolate in `requirements-vice-local.txt`.


## File e risorse

`gta.py` e l'avvio principale; `gta_menu.py`, `gta_camera.py` e
`gta_tracking.py` gestiscono menu, acquisizione e analisi locale del volto.
I moduli degli effetti mantengono il funzionamento precedente.

- `assets/`: stelle, mappa, audio WASTED e polizia, risorse VICE.
- `assets/gta6wallps/`: intera cartella di wallpaper trasferita dal progetto originale.
- `assets/benchmark_input.jpg`: immagine condivisa usata solo per il benchmark.
- `face_landmarker.task` e `models/`: copie locali dei modelli necessari.
- `.env`: configurazione OpenAI personale trasferita, esclusa da Git.
- `.venv/`: ambiente dell'app, escluso da Git.
- `.vice-local/`: Python, librerie GPU e cache dei modelli, esclusi da Git.

I percorsi delle risorse dipendono dalla cartella degli script, non dalla
directory di lavoro. Le componenti condivise sono copie autonome; non occorre
avviare Webcam Memes o altri progetti.

## Privacy

WASTED, WANTED e VICE Locale elaborano i fotogrammi localmente. VICE MODE AI
invia a OpenAI una singola foto soltanto dopo SPAZIO, usando la chiave
configurata. La webcam continua non viene inviata all'API. Foto e risultati
non vengono salvati automaticamente. Le app che ricevono OBS Virtual Camera
possono applicare proprie condizioni e politiche al video.

## Verifica

```powershell
.\.venv\Scripts\python.exe -B -m unittest discover -v
```

I test automatici non attivano la webcam, non riproducono audio e non usano
crediti API. I benchmark GPU sono separati dai test.
