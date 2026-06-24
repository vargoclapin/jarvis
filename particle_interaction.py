"""
3D Particle Interaction System — Python/Pygame + MediaPipe (nouvelle API ≥ 0.10)
Reproduit 粒子交互1_fixed.html avec contrôle webcam

Gestes :
  Main ouverte  →  rotation selon position de la paume
  Poing fermé   →  zoom selon distance caméra ↔ main

Clavier / souris (fallback) :
  C             →  couleur aléatoire
  F             →  plein écran
  Q / Esc       →  quitter


"""

import sys, math, random, threading
import numpy as np
import pygame

# ── Import optionnel ─────────────────────────────────────────────────────────
try:
    import cv2
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision
    from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions, RunningMode
    MEDIAPIPE_OK = True
except Exception as e:
    MEDIAPIPE_OK = False
    print(f"[WARN] MediaPipe indisponible : {e}")

# ── Constantes ───────────────────────────────────────────────────────────────
PARTICLE_COUNT       = 25_000
BASE_MODEL_SCALE     = 8.0
LERP_SPEED           = 0.07
FPS_CAP              = 60
W, H                 = 1280, 720
FOV                  = 700
INITIAL_COLOR        = (0, 255, 200)

MIN_GESTURE_SCALE    = 0.1
MAX_GESTURE_SCALE    = 2.4
PALM_SIZE_MIN        = 0.11
PALM_SIZE_MAX        = 0.28
PALM_SIZE_SCALE_ONE  = 0.15474

# ── Formes ───────────────────────────────────────────────────────────────────

def generate_tree(n):
    pos = np.zeros((n, 3), dtype=np.float32)
    for i in range(n):
        if i < n * 0.08:
            a = random.uniform(0, math.tau)
            r = (1.8 + abs(math.sin(a*5))*3.0) * math.sqrt(random.random())
            pos[i] = [r*math.cos(a), 20+random.uniform(-0.75,0.75), r*math.sin(a)*0.45]
        elif i < n * 0.92:
            h = random.uniform(-14, 18)
            ln = (h+14)/32; lp = (ln*4)%1.0
            br = (1-ln)*18*(lp**0.42)
            a  = random.uniform(0, math.tau)
            r  = max(0.0, br*math.sqrt(random.random())+random.uniform(-1.25,1.25))
            pos[i] = [math.cos(a)*r, h, math.sin(a)*r]
        else:
            h=random.uniform(-19,-14); r=random.uniform(0,2.2); a=random.uniform(0,math.tau)
            pos[i] = [math.cos(a)*r, h, math.sin(a)*r]
    return pos

def generate_saturn(n):
    pos = np.zeros((n, 3), dtype=np.float32)
    for i in range(n):
        if i < n * 0.42:
            theta=random.uniform(0,math.tau); phi=math.acos(2*random.random()-1)
            r=11.5*random.random()**(1/3)
            pos[i]=[r*math.sin(phi)*math.cos(theta), r*math.sin(phi)*math.sin(theta)*(0.82+random.uniform(0,0.06)), r*math.cos(phi)]
        else:
            rb=random.random()
            bb=14.5 if rb<0.22 else 18.5 if rb<0.48 else 23.5 if rb<0.72 else 28.5
            rr=bb+random.uniform(0,3.5)+math.sin(bb*1.7+i*0.013)*0.9
            tr=random.uniform(0,math.tau)
            rip=math.sin(tr*10+rr*0.9)*0.45+math.sin(tr*23)*0.18
            tk=random.uniform(-0.5,0.5)*(0.45+rb*0.85)
            pos[i]=[(rr+rip)*math.cos(tr), tk, (rr+rip)*math.sin(tr)]
    return pos

def generate_heart(n):
    pos = np.zeros((n, 3), dtype=np.float32)
    for i in range(n):
        t=random.uniform(0,math.tau); u=math.pi*random.uniform(-0.5,0.5)
        x=16*math.sin(t)**3
        y=13*math.cos(t)-5*math.cos(2*t)-2*math.cos(3*t)-math.cos(4*t)
        z=6*math.sin(u)*(1-abs(math.sin(t)))
        pos[i]=[x*1.35, y*1.35, z*1.35]
    return pos

# ── Projection ───────────────────────────────────────────────────────────────

def project(pts, rx, ry, zoom, cx, cy):
    cy_, sy_ = math.cos(ry), math.sin(ry)
    x1 =  pts[:,0]*cy_ + pts[:,2]*sy_
    z1 = -pts[:,0]*sy_ + pts[:,2]*cy_
    y1 =  pts[:,1]
    cx_, sx_ = math.cos(rx), math.sin(rx)
    y2 = y1*cx_ - z1*sx_
    z2 = y1*sx_ + z1*cx_
    fz = FOV / (FOV + z2*zoom*0.05)
    sx = (x1*zoom*fz + cx).astype(np.int32)
    sy = (-y2*zoom*fz + cy).astype(np.int32)
    return np.stack([sx,sy],axis=1), z2

# ── Mapping taille paume → échelle ──────────────────────────────────────────

def map_palm_to_scale(ps):
    ps = max(PALM_SIZE_MIN, min(PALM_SIZE_MAX, ps))
    if ps <= PALM_SIZE_SCALE_ONE:
        t = (ps - PALM_SIZE_MIN) / (PALM_SIZE_SCALE_ONE - PALM_SIZE_MIN)
        return MIN_GESTURE_SCALE + t*(1.0 - MIN_GESTURE_SCALE)
    t = (ps - PALM_SIZE_SCALE_ONE) / (PALM_SIZE_MAX - PALM_SIZE_SCALE_ONE)
    return 1.0 + t*(MAX_GESTURE_SCALE - 1.0)

# ── Thread webcam ─────────────────────────────────────────────────────────────

# URL du modèle MediaPipe Tasks (téléchargé une fois)
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
MODEL_PATH = "hand_landmarker.task"

def download_model():
    import urllib.request, os
    if not os.path.exists(MODEL_PATH):
        print("Téléchargement du modèle MediaPipe… ", end="", flush=True)
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("OK")

class HandTracker(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.target_rot_x    = 0.0
        self.target_rot_y    = 0.0
        self.target_scale    = 1.0
        self.hand_detected   = False
        self.is_fist         = False
        self.status_text     = "Démarrage…"
        self.thumb_frame     = None
        self._lock           = threading.Lock()
        self._running        = True

    def stop(self):
        self._running = False

    def _process_result(self, result, image, timestamp_ms):
        """Callback appelé par le HandLandmarker en mode LIVE_STREAM."""
        if not result.hand_landmarks:
            with self._lock:
                self.hand_detected = False
                self.target_scale  = 1.0
                self.status_text   = "Aucune main détectée"
            return

        lm = result.hand_landmarks[0]   # liste de NormalizedLandmark

        wrist       = lm[0]
        thumb_tip   = lm[4];  thumb_ip    = lm[3]
        index_pip   = lm[6];  index_tip   = lm[8]
        middle_pip  = lm[10]; middle_tip  = lm[12]; middle_base = lm[9]
        ring_pip    = lm[14]; ring_tip    = lm[16]
        pinky_pip   = lm[18]; pinky_tip   = lm[20]
        p17         = lm[17]; p5          = lm[5]

        palm_cx = (wrist.x + middle_base.x) * 0.5
        palm_cy = (wrist.y + middle_base.y) * 0.5
        palm_size = math.hypot(wrist.x - middle_base.x,
                               wrist.y - middle_base.y) or 0.08

        def td(tip):
            return math.hypot(tip.x - palm_cx, tip.y - palm_cy)

        def curled(tip, pip, f=1.08):
            return (math.hypot(tip.x-wrist.x, tip.y-wrist.y) <
                    math.hypot(pip.x-wrist.x, pip.y-wrist.y) * f)

        spread   = (td(thumb_tip)+td(index_tip)+td(middle_tip)+
                    td(ring_tip)+td(pinky_tip)) / 5
        openness = spread / palm_size

        thumb_curled  = td(thumb_tip) < td(thumb_ip) * 1.1
        curled_count  = sum([curled(index_tip,  index_pip),
                             curled(middle_tip, middle_pip),
                             curled(ring_tip,   ring_pip),
                             curled(pinky_tip,  pinky_pip, 1.12)])
        is_fist = openness < 1.72 and thumb_curled and curled_count >= 3

        hand_cx = (wrist.x + middle_base.x + p5.x + p17.x) * 0.25
        hand_cy = (wrist.y + middle_base.y + p5.y + p17.y) * 0.25

        # miroir horizontal (webcam retournée comme dans le HTML)
        tgt_ry = -((1.0 - hand_cx) - 0.5) * math.pi * 1.8
        tgt_rx =  (hand_cy - 0.5) * math.pi * 1.1

        scale = (max(MIN_GESTURE_SCALE,
                     min(MAX_GESTURE_SCALE, map_palm_to_scale(palm_size)))
                 if is_fist else 1.0)

        with self._lock:
            self.target_rot_x  = tgt_rx
            self.target_rot_y  = tgt_ry
            self.target_scale  = scale
            self.hand_detected = True
            self.is_fist       = is_fist
            self.status_text   = (f"{'Poing' if is_fist else 'Main ouverte'}"
                                  f" — zoom {scale:.2f}x")

    def run(self):
        if not MEDIAPIPE_OK:
            self.status_text = "MediaPipe non installé"
            return
        try:
            download_model()
        except Exception as e:
            self.status_text = f"Erreur téléchargement modèle : {e}"
            return

        try:
            options = HandLandmarkerOptions(
                base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
                running_mode=RunningMode.LIVE_STREAM,
                num_hands=1,
                min_hand_detection_confidence=0.5,
                min_hand_presence_confidence=0.5,
                min_tracking_confidence=0.5,
                result_callback=self._process_result,
            )
            landmarker = HandLandmarker.create_from_options(options)
        except Exception as e:
            self.status_text = f"Erreur init MediaPipe : {e}"
            return

        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            self.status_text = "Webcam introuvable"
            return

        self.status_text = "Webcam active — montrez votre main"
        ts = 0

        while self._running:
            ret, frame = cap.read()
            if not ret:
                continue

            # miniature pour l'affichage
            thumb = cv2.flip(cv2.resize(frame, (180, 135)), 1)
            self.thumb_frame = thumb

            # envoi au landmarker
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            ts += 33   # ~30 fps simulé en ms
            landmarker.detect_async(mp_image, ts)

        cap.release()
        landmarker.close()

# ── Application ───────────────────────────────────────────────────────────────

class App:
    MODEL_NAMES = {"tree":"Arbre de Noël (1)", "saturn":"Saturne (2)", "heart":"Cœur (3)"}

    def __init__(self):
        pygame.init()
        pygame.display.set_caption("3D Particle Interaction — Python + MediaPipe")
        self.screen = pygame.display.set_mode((W, H), pygame.RESIZABLE)
        self.clock  = pygame.time.Clock()
        self.font   = pygame.font.SysFont("monospace", 14)
        self.full   = False

        print("Génération des particules…", end=" ", flush=True)
        self.targets = {"tree": generate_tree(PARTICLE_COUNT),
                        "saturn": generate_saturn(PARTICLE_COUNT),
                        "heart": generate_heart(PARTICLE_COUNT)}
        print("OK")

        self.current_model    = "tree"
        self.positions        = self.targets["tree"].copy()
        self.rot_x            = 0.2
        self.rot_y            = 0.0
        self.zoom             = BASE_MODEL_SCALE
        self.gesture_scale    = 1.0
        self.tgt_gesture_scale= 1.0
        self.drag             = False
        self.last_mx = self.last_my = 0
        self.auto_rot         = 0.003
        self.color            = INITIAL_COLOR

        self.tracker = HandTracker() if MEDIAPIPE_OK else None
        if self.tracker:
            self.tracker.start()

    def run(self):
        while True:
            self.clock.tick(FPS_CAP)
            self.handle_events()
            self.update()
            self.draw()

    def handle_events(self):
        for e in pygame.event.get():
            if e.type == pygame.QUIT: self.quit()
            elif e.type == pygame.KEYDOWN:
                k = e.key
                if k in (pygame.K_ESCAPE, pygame.K_q): self.quit()
                elif k == pygame.K_1: self.current_model = "tree"
                elif k == pygame.K_2: self.current_model = "saturn"
                elif k == pygame.K_3: self.current_model = "heart"
                elif k == pygame.K_c:
                    self.color = tuple(random.randint(80,255) for _ in range(3))
                elif k == pygame.K_f: self.toggle_fullscreen()
            elif e.type == pygame.MOUSEBUTTONDOWN:
                if e.button == 1:
                    self.drag=True; self.last_mx,self.last_my=e.pos
                elif e.button == 4: self.zoom = min(self.zoom*1.08, 80.0)
                elif e.button == 5: self.zoom = max(self.zoom/1.08, 1.5)
            elif e.type == pygame.MOUSEBUTTONUP:
                if e.button == 1: self.drag = False
            elif e.type == pygame.MOUSEMOTION and self.drag:
                dx,dy = e.pos[0]-self.last_mx, e.pos[1]-self.last_my
                self.rot_y += dx*0.006; self.rot_x += dy*0.006
                self.last_mx,self.last_my = e.pos
            elif e.type == pygame.VIDEORESIZE:
                self.screen = pygame.display.set_mode(e.size, pygame.RESIZABLE)

    def update(self):
        target = self.targets[self.current_model]
        self.positions += (target - self.positions) * LERP_SPEED

        if self.tracker:
            with self.tracker._lock:
                detected = self.tracker.hand_detected
                trx = self.tracker.target_rot_x
                try_ = self.tracker.target_rot_y
                tsc = self.tracker.target_scale
            if detected:
                self.rot_y += (try_ - self.rot_y) * 0.05
                self.rot_x += (trx - self.rot_x) * 0.05
                self.tgt_gesture_scale = tsc
            elif not self.drag:
                self.rot_y += self.auto_rot
        elif not self.drag:
            self.rot_y += self.auto_rot

        self.gesture_scale += (self.tgt_gesture_scale - self.gesture_scale) * 0.12
        if not self.drag:
            self.zoom = BASE_MODEL_SCALE * self.gesture_scale

    def draw(self):
        sw, sh = self.screen.get_size()
        cx, cy = sw//2, sh//2

        ov = pygame.Surface((sw,sh), pygame.SRCALPHA)
        ov.fill((5,5,5,210))
        self.screen.blit(ov,(0,0))

        xy, z_depth = project(self.positions, self.rot_x, self.rot_y, self.zoom, cx, cy)
        order  = np.argsort(z_depth)
        z_min  = z_depth.min()
        z_range= max(z_depth.max()-z_min, 1e-6)

        surf = pygame.Surface((sw,sh)); surf.fill((0,0,0))
        r,g,b = self.color
        pix = pygame.PixelArray(surf)
        for idx in order:
            px,py = int(xy[idx,0]), int(xy[idx,1])
            if 0<=px<sw and 0<=py<sh:
                dt = (z_depth[idx]-z_min)/z_range
                al = 0.35+dt*0.65
                rr,gg,bb = min(255,int(r*al)), min(255,int(g*al)), min(255,int(b*al))
                pix[px,py] = (rr,gg,bb)
                if dt > 0.6:
                    dim=(rr//3,gg//3,bb//3)
                    for ddx,ddy in ((-1,0),(1,0),(0,-1),(0,1)):
                        nx_,ny_=px+ddx,py+ddy
                        if 0<=nx_<sw and 0<=ny_<sh:
                            pix[nx_,ny_]=dim
        del pix
        self.screen.blit(surf,(0,0),special_flags=pygame.BLEND_ADD)

        # miniature webcam
        if self.tracker and self.tracker.thumb_frame is not None:
            f = self.tracker.thumb_frame[:,:,::-1].copy()
            ts = pygame.surfarray.make_surface(np.transpose(f,(1,0,2)))
            tw,th = ts.get_size()
            px_,py_ = sw-tw-16, sh-th-16
            pygame.draw.rect(self.screen,self.color,(px_-2,py_-2,tw+4,th+4),2,6)
            self.screen.blit(ts,(px_,py_))

        # HUD
        st = self.tracker.status_text if self.tracker else "MediaPipe non installé"
        for i,line in enumerate([
            f"FPS : {self.clock.get_fps():.0f}   |   {self.MODEL_NAMES[self.current_model]}",
            f"Webcam : {st}",
            "1/2/3 → forme   C → couleur   F → plein écran   Q → quitter",
            "Main ouverte → rotation   |   Poing → zoom",
        ]):
            sh2 = self.font.render(line,True,(0,0,0))
            tx  = self.font.render(line,True,(200,200,200))
            self.screen.blit(sh2,(12,12+i*18+1))
            self.screen.blit(tx, (11,12+i*18))

        pygame.draw.circle(self.screen,self.color,(sw-30,30),14)
        pygame.draw.circle(self.screen,(255,255,255),(sw-30,30),14,2)
        pygame.display.flip()

    def toggle_fullscreen(self):
        self.full = not self.full
        self.screen = pygame.display.set_mode(
            (0,0) if self.full else (W,H),
            pygame.FULLSCREEN if self.full else pygame.RESIZABLE)

    def quit(self):
        if self.tracker: self.tracker.stop()
        pygame.quit(); sys.exit()

if __name__ == "__main__":
    App().run()