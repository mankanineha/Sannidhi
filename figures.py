"""Generate the 9 patent drawing sheets (FIG. 1 - FIG. 9) as black-and-white SVG."""
import os
from html import escape

W, H = 700, 940
OUT = os.path.dirname(os.path.abspath(__file__))

class Fig:
    def __init__(self, n):
        self.n = n
        self.el = []

    def add(self, s):
        self.el.append(s)

    def save(self):
        svg = (
            '<svg xmlns="http://www.w3.org/2000/svg" width="1400" height="1880" '
            'viewBox="0 0 %d %d" font-family="Arial, Helvetica, sans-serif" '
            'font-size="13" fill="black">\n'
            '<defs>\n'
            '<marker id="arr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" '
            'markerHeight="8" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="black"/></marker>\n'
            '<pattern id="hatch" width="6" height="6" patternUnits="userSpaceOnUse" '
            'patternTransform="rotate(45)"><line x1="0" y1="0" x2="0" y2="6" stroke="black" stroke-width="2"/></pattern>\n'
            '</defs>\n<rect width="%d" height="%d" fill="white"/>\n%s\n'
            '<text x="350" y="%d" text-anchor="middle" font-size="22" font-weight="bold">FIG. %s</text>\n</svg>\n'
        ) % (W, H, W, H, "\n".join(self.el), H - 18, self.n)
        with open(os.path.join(OUT, "fig%s.svg" % self.n), "w", encoding="utf-8") as fh:
            fh.write(svg)

def T(f, x, y, s, anchor="middle", size=13, bold=False):
    w = ' font-weight="bold"' if bold else ""
    f.add('<text x="%s" y="%s" text-anchor="%s" font-size="%s"%s>%s</text>' % (x, y, anchor, size, w, escape(s)))

def line(f, pts, dash=False, sw=1.5):
    d = ' stroke-dasharray="6 4"' if dash else ""
    f.add('<polyline points="%s" fill="none" stroke="black" stroke-width="%s"%s/>' %
          (" ".join("%s,%s" % p for p in pts), sw, d))

def arrow(f, pts, both=False, dash=False, sw=1.5):
    d = ' stroke-dasharray="6 4"' if dash else ""
    ms = ' marker-start="url(#arr)"' if both else ""
    f.add('<polyline points="%s" fill="none" stroke="black" stroke-width="%s"%s marker-end="url(#arr)"%s/>' %
          (" ".join("%s,%s" % p for p in pts), sw, d, ms))

def lab(f, x, y, w, h, text, pos="top"):
    """Reference numeral with a short lead line."""
    if pos == "top":
        line(f, [(x + w - 12, y), (x + w - 20, y - 9)], sw=1)
        T(f, x + w - 22, y - 11, text, "end", 13)
    elif pos == "r":
        line(f, [(x + w, y + 10), (x + w + 9, y + 3)], sw=1)
        T(f, x + w + 11, y + 7, text, "start", 13)
    elif pos == "l":
        line(f, [(x, y + 10), (x - 9, y + 3)], sw=1)
        T(f, x - 11, y + 7, text, "end", 13)

def box(f, x, y, w, h, lines, label=None, pos="top", dash=False, fill="white", size=13, sw=1.5):
    d = ' stroke-dasharray="8 5"' if dash else ""
    f.add('<rect x="%s" y="%s" width="%s" height="%s" fill="%s" stroke="black" stroke-width="%s"%s/>' %
          (x, y, w, h, fill, sw, d))
    n = len(lines)
    y0 = y + h / 2 - (n - 1) * (size + 3) / 2 + size * 0.36
    for i, s in enumerate(lines):
        T(f, x + w / 2, y0 + i * (size + 3), s, size=size)
    if label:
        lab(f, x, y, w, h, label, pos)

def frame(f, x, y, w, h, title_lines, label=None, pos="top", dash=False):
    d = ' stroke-dasharray="8 5"' if dash else ""
    f.add('<rect x="%s" y="%s" width="%s" height="%s" fill="none" stroke="black" stroke-width="1.5"%s/>' % (x, y, w, h, d))
    for i, s in enumerate(title_lines):
        T(f, x + w / 2, y + 20 + i * 16, s)
    if label:
        lab(f, x, y, w, h, label, pos)

def diamond(f, cx, cy, w, h, lines):
    f.add('<polygon points="%s,%s %s,%s %s,%s %s,%s" fill="white" stroke="black" stroke-width="1.5"/>' %
          (cx, cy - h / 2, cx + w / 2, cy, cx, cy + h / 2, cx - w / 2, cy))
    n = len(lines)
    y0 = cy - (n - 1) * 8 + 5
    for i, s in enumerate(lines):
        T(f, cx, y0 + i * 16, s)

def oval(f, cx, cy, rx, ry, lines):
    f.add('<ellipse cx="%s" cy="%s" rx="%s" ry="%s" fill="white" stroke="black" stroke-width="1.5"/>' % (cx, cy, rx, ry))
    n = len(lines)
    y0 = cy - (n - 1) * 8 + 5
    for i, s in enumerate(lines):
        T(f, cx, y0 + i * 16, s)

def bh(lines):
    return max(36, len(lines) * 16 + 22)

# FIG 1
def fig1():
    f = Fig(1)
    f.add('<rect x="8" y="40" width="684" height="860" fill="none" stroke="black" stroke-width="1.5" stroke-dasharray="10 6"/>')
    lab(f, 8, 40, 684, 860, "100", "top")
    box(f, 20, 109, 130, 70, ["Wi-Fi sensing", "node (ESP32)"], "101", "r")
    line(f, [(85, 109), (85, 84)])
    line(f, [(73, 70), (85, 84), (97, 70)])
    line(f, [(98, 80), (108, 74)], sw=1)
    T(f, 110, 76, "102", "start")
    box(f, 20, 250, 130, 60, ["Wi-Fi access", "point"], "103", "r")
    arrow(f, [(85, 179), (76, 195), (94, 210), (76, 225), (94, 238), (85, 250)], both=True)
    frame(f, 240, 72, 250, 533, ["Edge processing unit", "(Raspberry Pi 5)"], "104")
    names = [("110", ["Frame receiver"]), ("111", ["Windowing and", "normalisation module"]),
             ("112", ["Inference module", "(FallGuardNet)"]), ("113", ["Decision module"]),
             ("114", ["Agentic conversational", "layer"]), ("115", ["Escalation module"])]
    ys = []
    for i, (n, ls) in enumerate(names):
        y = 120 + i * 78
        ys.append(y)
        box(f, 265, y, 200, 48, ls, n, "top")
        if i:
            arrow(f, [(365, ys[i - 1] + 48), (365, y)])
    arrow(f, [(150, 144), (265, 144)])
    T(f, 207, 136, "UDP")
    frame(f, 20, 390, 175, 140, ["Voice interface"], "105")
    box(f, 28, 424, 105, 42, ["Microphone"], "105a", "r")
    box(f, 28, 478, 105, 42, ["Speaker"], "105b", "r")
    arrow(f, [(195, 456), (265, 456)], both=True)
    box(f, 535, 420, 140, 70, ["Cloud database"], "106", "top")
    arrow(f, [(465, 456), (535, 456)], both=True)
    arrow(f, [(465, 534), (615, 534), (615, 490)])
    box(f, 230, 650, 270, 48, ["Messaging service"], "109", "top")
    arrow(f, [(365, 558), (365, 650)])
    frame(f, 230, 745, 270, 130, ["Caregiver device"], "108")
    box(f, 270, 785, 170, 60, ["Caregiver", "application"], "107", "r")
    arrow(f, [(365, 698), (365, 745)])
    arrow(f, [(665, 490), (665, 810), (500, 810)])
    f.save()

# FIG 2 (Room Deployment)
def fig2():
    f = Fig(2)
    f.add('<g transform="translate(0,120)">')
    floor = 550
    f.add('<rect x="40" y="80" width="620" height="470" fill="none" stroke="black" stroke-width="3"/>')
    lab(f, 40, 80, 620, 470, "202", "top")
    
    # sofa
    f.add('<rect x="130" y="505" width="160" height="45" fill="white" stroke="black" stroke-width="1.5"/>')
    f.add('<rect x="130" y="465" width="160" height="42" fill="white" stroke="black" stroke-width="1.5"/>')
    f.add('<rect x="113" y="490" width="18" height="60" fill="white" stroke="black" stroke-width="1.5"/>')
    f.add('<rect x="289" y="490" width="18" height="60" fill="white" stroke="black" stroke-width="1.5"/>')
    
    # table with edge unit 104 and voice interface 105 (wider layout)
    f.add('<rect x="500" y="460" width="160" height="8" fill="white" stroke="black" stroke-width="1.5"/>')
    f.add('<rect x="510" y="468" width="8" height="82" fill="white" stroke="black" stroke-width="1.5"/>')
    f.add('<rect x="642" y="468" width="8" height="82" fill="white" stroke="black" stroke-width="1.5"/>')
    # single enclosure on table holding edge unit 104 and voice interface 105
    f.add('<rect x="504" y="392" width="152" height="68" fill="white" stroke="black" stroke-width="2"/>')
    box(f, 508, 396, 70, 60, ["Edge", "unit"], size=12)
    box(f, 582, 396, 70, 60, ["Voice", "interface"], size=11)
    # clean lead lines for 104 and 105 above enclosure
    line(f, [(543, 375), (543, 396)], sw=1)
    T(f, 543, 368, "104")
    line(f, [(617, 375), (617, 396)], sw=1)
    T(f, 617, 368, "105")
    
    # shelves, 101 with antenna 102, 103
    f.add('<rect x="40" y="250" width="100" height="8" fill="white" stroke="black" stroke-width="1.5"/>')
    box(f, 52, 212, 70, 38, ["Sensing", "node"], size=12)
    line(f, [(60, 212), (54, 205)], sw=1)
    T(f, 46, 200, "101", "start")
    line(f, [(87, 212), (87, 188)])
    line(f, [(76, 176), (87, 188), (98, 176)])
    line(f, [(98, 182), (110, 176)], sw=1)
    T(f, 112, 175, "102", "start")
    f.add('<rect x="560" y="250" width="100" height="8" fill="white" stroke="black" stroke-width="1.5"/>')
    box(f, 575, 215, 70, 35, ["Access", "point"], size=12)
    line(f, [(612, 215), (612, 207)], sw=1)
    T(f, 612, 203, "103")
    
    # radio path 203 bouncing off fallen person to sensing antenna
    line(f, [(575, 235), (380, 532), (87, 192)], dash=True, sw=1)
    line(f, [(275, 415), (315, 435)], sw=1)
    T(f, 270, 412, "203", "end")
    
    # resident fallen 201 (solid outline, only person in room)
    f.add('<circle cx="355" cy="535" r="14" fill="white" stroke="black" stroke-width="1.5"/>')
    f.add('<polyline points="369,535 432,535" fill="none" stroke="black" stroke-width="1.5"/>')
    f.add('<polyline points="380,535 398,524 418,524" fill="none" stroke="black" stroke-width="1.5"/>')
    f.add('<polyline points="380,535 398,545 418,545" fill="none" stroke="black" stroke-width="1.5"/>')
    f.add('<polyline points="432,535 456,527 482,530" fill="none" stroke="black" stroke-width="1.5"/>')
    f.add('<polyline points="432,535 456,545 482,545" fill="none" stroke="black" stroke-width="1.5"/>')
    # clean lead line for 201 without numeral collision
    line(f, [(335, 498), (355, 521)], sw=1)
    T(f, 330, 495, "201", "end")
    
    f.add("</g>")
    f.save()

# FIG 3
def fig3():
    f = Fig(3)
    steps = [
        ("S301", ["Connect to access point as station;", "disable Wi-Fi power saving"]),
        ("S302", ["Enable CSI reporting;", "register CSI callback"]),
        ("S303", ["Receive data frame;", "callback invoked"]),
        ("S304", ["Read 64 complex (I, Q) samples"]),
        ("S305", ["Compute amplitude per sample:", "square root of (I\u00b2 + Q\u00b2)"]),
        ("S306", ["Keep 52 active subcarriers (bins 38\u201363", "and 1\u201326); if first word invalid,", "replace bin 1 by bin 2"]),
        ("S307", ["Form frame (timestamp, frame number,", "52 amplitudes); put on bounded queue"]),
        ("S308", ["Writer task dequeues frame"]),
        ("S309", ["Transmit frame as UDP datagram to", "edge processing unit"]),
    ]
    cx, w, y, gap = 350, 380, 30, 30
    pos = {}
    prev = None
    for sid, ls in steps:
        h = bh(ls)
        box(f, cx - w / 2, y, w, h, ls, sid, "top")
        if prev:
            arrow(f, [(cx, prev[0] + prev[1]), (cx, y)])
        pos[sid] = (y, h)
        prev = (y, h)
        y += h + gap
    y9, h9 = pos["S309"]
    y3, h3 = pos["S303"]
    arrow(f, [(cx + w / 2, y9 + h9 / 2), (595, y9 + h9 / 2), (595, y3 + h3 / 2), (cx + w / 2, y3 + h3 / 2)])
    T(f, 630, (y9 + y3) / 2, "next frame")
    f.save()

# FIG 4
def fig4():
    f = Fig(4)
    cx, w, gap = 345, 240, 30
    items = [
        ("box", ["Receive frame;", "validate 52 values"], "S401", 46),
        ("box", ["Append frame to buffer"], "S402", 36),
        ("dia", ["Buffer holds", "950 frames?"], "S403", 84),
        ("box", ["Replace non-finite", "values by zero"], "S404", 46),
        ("box", ["Global z-score normalisation", "of 950 \u00d7 52 window"], "S405", 46),
        ("box", ["Run FallGuardNet;", "obtain two logits"], "S406", 46),
        ("box", ["Softmax; fall probability p"], "S407", 36),
        ("dia", ["p \u2265 T2 ?"], "S408", 84),
        ("dia", ["p \u2265 T1, or motion", "measure elevated?"], "S411", 84),
        ("box", ["State: NON-FALL;", "clear latch"], "S413", 46),
    ]
    y = 25
    P = {}
    prev = None
    for kind, ls, sid, h in items:
        if kind == "box":
            box(f, cx - w / 2, y, w, h, ls, sid, "top")
        else:
            diamond(f, cx, y + h / 2, w, h, ls)
            T(f, cx + w / 2 - (25 if sid == "S408" else -6), y + h / 2 - (36 if sid == "S408" else 12), sid, "start")
        if prev:
            arrow(f, [(cx, prev[0] + prev[1]), (cx, y)])
        P[sid] = (y, h)
        prev = (y, h)
        y += h + gap
    c = lambda s: P[s][0] + P[s][1] / 2
    T(f, cx + 8, P["S403"][0] + P["S403"][1] + 20, "Yes", "start", 12)
    T(f, cx + 8, P["S408"][0] + P["S408"][1] + 20, "No", "start", 12)
    T(f, cx + 8, P["S411"][0] + P["S411"][1] + 20, "No", "start", 12)
    line(f, [(cx - w / 2, c("S403")), (18, c("S403"))])
    T(f, cx - w / 2 - 6, c("S403") - 6, "No", "end", 12)
    arrow(f, [(cx - w / 2, c("S413")), (18, c("S413")), (18, c("S401")), (cx - w / 2, c("S401"))])
    y8 = c("S408")
    arrow(f, [(cx + w / 2, y8), (515, y8)])
    T(f, 490, y8 - 6, "Yes", "middle", 12)
    box(f, 515, y8 - 23, 175, 46, ["State:", "CONFIRMED FALL"], "S409", "top")
    y10 = y8 + 23 + gap
    box(f, 515, y10, 175, 46, ["Send event once to", "agentic conversational", "layer 114"], "S410", "top")
    arrow(f, [(602, y8 + 23), (602, y10)])
    oval(f, 602, y10 + 46 + gap + 20, 55, 20, ["To FIG. 8 (S801)"])
    arrow(f, [(602, y10 + 46), (602, y10 + 46 + gap)])
    y11 = c("S411")
    arrow(f, [(cx - w / 2, y11), (185, y11)])
    T(f, 205, y11 - 6, "Yes", "middle", 12)
    box(f, 45, y11 - 23, 140, 46, ["State:", "POSSIBLE FALL"], "S412", "top")
    line(f, [(45, y11), (18, y11)])
    f.save()

# FIG 5
def fig5():
    f = Fig(5)
    box(f, 130, 30, 300, 40, ["Normalised window 950 \u00d7 52"], "501", "l")
    box(f, 130, 100, 300, 40, ["Transpose to 52 \u00d7 950"], "502", "l")
    arrow(f, [(280, 70), (280, 100)])
    line(f, [(280, 140), (280, 160)])
    line(f, [(100, 160), (460, 160)])
    for x in (100, 280, 460):
        arrow(f, [(x, 160), (x, 192)])
    br = [("503", 20, ["Fine branch:", "k = 3,", "dilations 1, 2, 4"]),
          ("504", 200, ["Medium branch:", "k = 7,", "dilations 1, 2, 4"]),
          ("505", 380, ["Coarse branch:", "k = 15,", "dilations 1, 2"])]
    for n, x, ls in br:
        box(f, x, 192, 160, 62, ls, n, "top")
        line(f, [(x + 80, 254), (x + 80, 278)])
    line(f, [(100, 278), (460, 278)])
    arrow(f, [(280, 278), (280, 296)])
    box(f, 130, 296, 300, 40, ["Align and concatenate"], "506", "l")
    arrow(f, [(280, 336), (280, 366)])
    box(f, 130, 366, 300, 50, ["Fusion layer:", "1 \u00d7 1 convolution, BN, GELU"], "507", "l")
    arrow(f, [(280, 416), (280, 446)])
    box(f, 130, 446, 300, 50, ["Channel attention", "(squeeze-and-excitation)"], "508", "l")
    line(f, [(280, 496), (280, 518)])
    line(f, [(90, 518), (280, 518)])
    arrow(f, [(90, 518), (90, 548)])
    arrow(f, [(265, 518), (265, 548)])
    box(f, 15, 548, 150, 50, ["Temporal attention", "pooling"], "509", "top")
    box(f, 190, 548, 150, 50, ["Max pooling", "over time"], "510", "top")
    box(f, 365, 548, 150, 50, ["Periodicity", "score"], "511", "top")
    box(f, 540, 548, 150, 50, ["Stillness", "score"], "512", "top")
    line(f, [(430, 50), (615, 50), (615, 504)])
    arrow(f, [(615, 504), (615, 548)])
    arrow(f, [(615, 504), (440, 504), (440, 548)])
    for cx in (90, 265, 440, 615):
        line(f, [(cx, 598), (cx, 620)])
    line(f, [(90, 620), (615, 620)])
    arrow(f, [(330, 620), (330, 640)])
    box(f, 180, 640, 300, 40, ["Feature concatenation"], "513", "l")
    arrow(f, [(330, 680), (330, 710)])
    box(f, 180, 710, 300, 40, ["Classifier head"], "514", "l")
    arrow(f, [(330, 750), (330, 780)])
    box(f, 180, 780, 300, 50, ["Output logits:", "non-fall, fall"], "515", "l")
    f.save()

# FIG 6
def fig6():
    f = Fig(6)
    box(f, 225, 20, 260, 46, ["Labelled CSI", "amplitude windows"], "601", "top")
    box(f, 225, 95, 260, 46, ["Augmentation (time reversal,", "shift, scaling, noise)"], "602", "top")
    box(f, 225, 170, 260, 36, ["Class-weighted sampler"], "603", "top")
    arrow(f, [(355, 66), (355, 95)])
    arrow(f, [(355, 141), (355, 170)])
    line(f, [(355, 206), (355, 228)])
    arrow(f, [(355, 228), (355, 250)])
    arrow(f, [(355, 228), (115, 228), (115, 250)])
    box(f, 20, 250, 190, 46, ["Teacher network", "(fixed)"], "604", "top")
    box(f, 260, 250, 190, 46, ["Student network", "(FallGuardNet)"], "606", "top")
    box(f, 20, 340, 190, 46, ["Teacher softened", "outputs"], "605", "top")
    box(f, 260, 340, 190, 46, ["Student softened", "outputs"], "607", "top")
    arrow(f, [(115, 296), (115, 340)])
    arrow(f, [(355, 296), (355, 340)])
    box(f, 520, 340, 170, 62, ["Hard-label loss", "(class-weighted", "cross-entropy)"], "609", "top")
    arrow(f, [(450, 262), (560, 262), (560, 340)])
    arrow(f, [(485, 188), (620, 188), (620, 340)])
    T(f, 626, 270, "labels", "start", 12)
    box(f, 80, 450, 340, 56, ["Distillation loss", "(KL divergence, T = 4)"], "608", "top")
    arrow(f, [(115, 386), (115, 450)])
    arrow(f, [(355, 386), (355, 450)])
    box(f, 150, 570, 300, 36, ["Combined loss: \u03b1 \u00d7 608 + (1 \u2212 \u03b1) \u00d7 609"], "610", "top")
    arrow(f, [(230, 506), (230, 570)])
    arrow(f, [(605, 402), (605, 588), (450, 588)])
    box(f, 150, 650, 300, 36, ["Optimiser"], "611", "top")
    arrow(f, [(300, 606), (300, 650)])
    arrow(f, [(450, 668), (480, 668), (480, 285), (450, 285)], dash=True)
    T(f, 488, 480, "weight", "start", 12)
    T(f, 488, 496, "update", "start", 12)
    box(f, 150, 730, 300, 36, ["Trained student network"], "612", "top")
    arrow(f, [(300, 686), (300, 730)])
    box(f, 150, 800, 300, 46, ["Export to ONNX;", "deploy on edge unit"], "613", "top")
    arrow(f, [(300, 766), (300, 800)])
    f.save()

# FIG 7 (Agentic conversational layer)
def fig7():
    f = Fig(7)
    frame(f, 160, 60, 380, 560, ["Agentic conversational layer"], "114")
    
    # external top
    box(f, 260, 15, 180, 32, ["Decision module"], "113", "l")
    arrow(f, [(350, 47), (350, 85)])
    
    # internal
    box(f, 260, 85, 180, 32, ["Event interface"], "701", "r")
    arrow(f, [(350, 117), (350, 145)])
    
    box(f, 260, 145, 180, 32, ["Incident manager"], "702", "top")
    arrow(f, [(315, 177), (255, 205)])
    arrow(f, [(385, 177), (445, 205)])
    
    box(f, 180, 205, 150, 32, ["Incident store"], "703", "top")
    box(f, 360, 205, 170, 32, ["Conversation state machine"], "704", "top")
    
    # speech synthesis
    box(f, 360, 265, 170, 32, ["Speech synthesis engine"], "705", "top")
    arrow(f, [(445, 237), (445, 265)])
    
    # external speaker
    box(f, 575, 265, 105, 32, ["Speaker"], "105b", "top")
    arrow(f, [(530, 281), (575, 281)])
    
    # Scheduler
    box(f, 180, 265, 150, 32, ["Scheduler (optional)"], "710", "top")
    arrow(f, [(330, 281), (360, 281)]) # into 704
    
    # external microphone
    box(f, 20, 335, 105, 32, ["Microphone"], "105a", "top")
    arrow(f, [(125, 351), (180, 351)])
    
    box(f, 180, 335, 150, 32, ["Voice activity detector"], "706", "top")
    arrow(f, [(330, 351), (365, 351)])
    
    box(f, 365, 335, 165, 32, ["Speech-to-text engine"], "707", "top")
    arrow(f, [(447, 367), (447, 395)])
    
    box(f, 365, 395, 165, 42, ["Interpretation module", "(language model)"], "708", "top")
    arrow(f, [(530, 416), (545, 416), (545, 221), (530, 221)]) # back to 704
    
    box(f, 260, 480, 180, 34, ["Safety-policy controller", "(rule-based)"], "709", "top")
    arrow(f, [(380, 237), (380, 480)]) # from 704
    
    # external bottom
    arrow(f, [(350, 514), (350, 660)])
    T(f, 360, 580, "escalate", "start")
    
    box(f, 250, 660, 200, 34, ["Escalation module"], "115", "l")
    arrow(f, [(350, 694), (350, 725)])
    
    box(f, 250, 725, 200, 34, ["Messaging service"], "109", "l")
    arrow(f, [(350, 759), (350, 790)])
    
    box(f, 230, 790, 240, 36, ["Caregiver device (application 107)"], "108", "l")
    arrow(f, [(350, 826), (350, 826)])
    
    f.save()

# FIG 8
def fig8():
    f = Fig(8)
    cx, w = 290, 230
    items = [
        ("box", ["Decision module sends event", "to conversational layer"], "S801", 36),
        ("box", ["Create incident; retrieve context"], "S802", 30),
        ("box", ["State machine starts workflow"], "S803", 30),
        ("box", ["Speak prompt in preferred language"], "S804", 36),
        ("box", ["Capture response"], "S805", 30),
        ("box", ["Detect speech"], "S806", 30),
        ("box", ["Transcribe audio"], "S807", 30),
        ("box", ["Interpret response; recognised intent"], "S808", 36),
        ("dia", ["Safety-policy controller", "evaluates state and intent"], "S809", 64),
    ]
    y = 20
    gap = 14
    P = {}
    prev = None
    for kind, ls, sid, h in items:
        if kind == "box":
            box(f, cx - w / 2, y, w, h, ls, sid, "top")
        else:
            diamond(f, cx, y + h / 2, 220, h, ls)
            T(f, cx - 120, y + h / 2 - 20, sid, "end")
        if prev:
            arrow(f, [(cx, prev[0] + prev[1]), (cx, y)])
        P[sid] = (y, h)
        prev = (y, h)
        y += h + gap

    y9 = P["S809"][0] + P["S809"][1] / 2
    
    # 4 branches from S809 (cy = y9)
    # 1. Left: S810 (Clearly okay)
    box(f, 15, y9 + 60, 120, 40, ["Apply confirmation", "policy"], "S810", "top")
    line(f, [(cx - 110, y9), (75, y9)])
    arrow(f, [(75, y9), (75, y9 + 60)])
    T(f, 120, y9 - 6, "Clearly okay", "middle", 11)
    
    # 2. Mid-Left: S811 (Unclear response)
    box(f, 145, y9 + 60, 110, 40, ["Ask clarifying", "question"], "S811", "top")
    line(f, [(cx - 55, y9 + 16), (200, y9 + 40)])
    arrow(f, [(200, y9 + 40), (200, y9 + 60)])
    T(f, 175, y9 + 46, "Unclear response", "end", 11)
    # S811 back to S805
    arrow(f, [(200, y9 + 100), (200, y9 + 130), (8, y9 + 130), (8, P["S805"][0] + 15), (cx - w / 2, P["S805"][0] + 15)])
    
    # 3. Mid-Right: S812 (No usable response)
    box(f, 320, y9 + 60, 115, 40, ["Wait for configured", "timeout"], "S812", "top")
    line(f, [(cx + 55, y9 + 16), (375, y9 + 40)])
    arrow(f, [(375, y9 + 40), (375, y9 + 60)])
    T(f, 395, y9 + 46, "No usable response", "start", 11)
    
    # S813 Diamond
    diamond(f, 375, y9 + 155, 115, 50, ["Response", "received?"])
    T(f, 310, y9 + 140, "S813", "end")
    arrow(f, [(375, y9 + 100), (375, y9 + 130)])
    
    # S813 Yes -> back to S805
    arrow(f, [(317, y9 + 155), (15, y9 + 155), (15, P["S805"][0] + 15), (cx - w / 2, P["S805"][0] + 15)])
    T(f, 220, y9 + 147, "Response received", "middle", 11)
    
    # 4. Far-Right: S814 (Needs help or reports pain)
    box(f, 470, y9 + 60, 145, 40, ["Trigger caregiver", "escalation"], "S814", "top")
    line(f, [(cx + 110, y9), (540, y9)])
    arrow(f, [(540, y9), (540, y9 + 60)])
    T(f, 470, y9 - 6, "Needs help or reports pain", "middle", 11)
    
    # S813 No -> S815
    arrow(f, [(432, y9 + 155), (540, y9 + 155), (540, y9 + 175)])
    T(f, 475, y9 + 147, "Still no response", "middle", 11)
    
    # S814 -> S815 -> S816
    box(f, 470, y9 + 175, 145, 40, ["Send caregiver", "notification"], "S815", "top")
    arrow(f, [(540, y9 + 100), (540, y9 + 175)])
    
    box(f, 470, y9 + 235, 145, 40, ["Track delivery and", "acknowledgement"], "S816", "top")
    arrow(f, [(540, y9 + 215), (540, y9 + 235)])
    
    # S810 and S816 -> S817
    box(f, 190, y9 + 270, 220, 36, ["Save outcome;", "update incident store"], "S817", "top")
    arrow(f, [(75, y9 + 100), (75, y9 + 288), (190, y9 + 288)])
    arrow(f, [(540, y9 + 275), (540, y9 + 288), (410, y9 + 288)])
    
    # S818 (Scheduled check-in)
    oval(f, 545, 45, 115, 22, ["Scheduled check-in or", "medication reminder"])
    T(f, 545, 18, "S818")
    arrow(f, [(430, 45), (cx, 45), (cx, P["S802"][0])], dash=True)
    
    f.save()

# FIG 9
def fig9():
    f = Fig(9)
    # (a) confusion matrix
    T(f, 20, 35, "(a)", "start", 15, True)
    cw, ch, x0, y0 = 130, 70, 270, 80
    T(f, x0 + cw / 2, y0 - 12, "Predicted non-fall", size=13)
    T(f, x0 + cw * 1.5, y0 - 12, "Predicted fall", size=13)
    T(f, x0 - 10, y0 + ch / 2 + 5, "Actual non-fall", "end", 13)
    T(f, x0 - 10, y0 + ch * 1.5 + 5, "Actual fall", "end", 13)
    vals = [[60, 0], [2, 8]]
    for r in range(2):
        for c in range(2):
            fill = "url(#hatch)" if r == c else "white"
            f.add('<rect x="%s" y="%s" width="%s" height="%s" fill="white" stroke="black" stroke-width="1.5"/>' %
                  (x0 + c * cw, y0 + r * ch, cw, ch))
            if r == c:
                f.add('<rect x="%s" y="%s" width="%s" height="%s" fill="%s"/>' %
                      (x0 + c * cw + 2, y0 + r * ch + 2, cw - 4, ch - 4, fill))
            f.add('<rect x="%s" y="%s" width="44" height="30" fill="white"/>' % (x0 + c * cw + cw / 2 - 22, y0 + r * ch + ch / 2 - 18))
            T(f, x0 + c * cw + cw / 2, y0 + r * ch + ch / 2 + 8, str(vals[r][c]), size=24, bold=True)
            
    MODELS = ["MLP", "CNN", "GRU", "LSTM", "Transformer", "MobileNetV3",
              "FallGuardNet first version", "FallGuardNet"]
    F1 = [0.455, 0.562, 0.476, 0.321, 0.667, 0.750, 0.667, 0.889]
    SPEC = [0.883, 0.783, 0.900, 0.383, 0.867, 0.917, 0.867, 1.000]
    
    def barchart(f, y0, tag, xlabel, vals):
        x0, sc, row = 215, 430, 22
        T(f, 20, y0, tag, "start", 15, True)
        T(f, 20, y0 + 18, "Model", "start", 13)
        top = y0 + 28
        for i, (m, v) in enumerate(zip(MODELS, vals)):
            y = top + i * row
            T(f, x0 - 8, y + 14, m, "end", 13)
            fill = "url(#hatch)" if m == "FallGuardNet" else "white"
            f.add('<rect x="%s" y="%s" width="%s" height="14" fill="%s" stroke="black" stroke-width="1.2"/>' % (x0, y + 3, v * sc, fill))
            T(f, x0 + v * sc + 6, y + 14, "%.3f" % v, "start", 12)
        bot = top + len(MODELS) * row
        line(f, [(x0, top), (x0, bot), (x0 + sc + 30, bot)])
        for k in range(6):
            x = x0 + k * 0.2 * sc
            line(f, [(x, bot), (x, bot + 5)], sw=1)
            T(f, x, bot + 20, "%.1f" % (k * 0.2), size=12)
        T(f, x0 + sc / 2, bot + 40, xlabel, size=13)
        
    barchart(f, 285, "(b)", "F1 score", F1)
    barchart(f, 575, "(c)", "Specificity", SPEC)
    f.save()

if __name__ == "__main__":
    for fn in (fig1, fig2, fig3, fig4, fig5, fig6, fig7, fig8, fig9):
        fn()
    print("Wrote fig1.svg ... fig9.svg to", OUT)
