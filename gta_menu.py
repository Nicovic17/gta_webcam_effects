"""GTA-only selection, before opening webcam or GPU."""
import cv2
import numpy as np

EFFECTS = (
    ('wasted', 'WASTED Webcam', 'SPAZIO: slow motion, WASTED e respawn'),
    ('wanted', 'WANTED Webcam', 'SPAZIO: stelle, mappa, scansione e foto WANTED'),
    ('vice', 'VICE MODE AI Webcam - in progress', 'SPAZIO: foto GTA con la tua API key OpenAI'),
    ('vice_local', 'VICE Locale - GPU - in progress', 'Prompt libero: scrivi come trasformare la webcam'),
)
MENU_ROW_STEP = 108
MENU_CARD_HEIGHT = 88

def choose_effect():
    """Select before opening any camera, detector or audio device."""
    title = "GTA Webcam - Scegli effetto"
    selected = [None]
    menu_height = 146 + len(EFFECTS)*MENU_ROW_STEP
    image = np.full((menu_height,720,3),(26,29,34),np.uint8)
    cv2.putText(image,"GTA WEBCAM EFFECTS",(55,62),
                cv2.FONT_HERSHEY_SIMPLEX,.87,(240,245,247),2,cv2.LINE_AA)
    for index,(_,label,detail) in enumerate(EFFECTS):
        top = 100 + index*MENU_ROW_STEP
        cv2.rectangle(image,(50,top),(670,top+MENU_CARD_HEIGHT),(48,54,62),-1)
        cv2.rectangle(image,(50,top),(670,top+MENU_CARD_HEIGHT),(111,142,153),2)
        cv2.putText(image,f"{index+1}  {label}",(75,top+24),
                    cv2.FONT_HERSHEY_SIMPLEX,.64,(240,245,247),2,cv2.LINE_AA)
        cv2.putText(image,detail,(75,top+43),
                    cv2.FONT_HERSHEY_SIMPLEX,.43,(179,195,201),1,cv2.LINE_AA)
    keys = ', '.join(str(index+1) for index in range(len(EFFECTS)))
    cv2.putText(image,f"Clicca una voce o premi {keys}.  Q / ESC: esci.",(50,menu_height-20),
                cv2.FONT_HERSHEY_SIMPLEX,.50,(179,195,201),1,cv2.LINE_AA)

    def on_mouse(event,x,y,flags,userdata):
        if event == cv2.EVENT_LBUTTONUP and 50 <= x <= 670:
            for index,(mode,_,_) in enumerate(EFFECTS):
                if 100+index*MENU_ROW_STEP <= y <= 100+index*MENU_ROW_STEP+MENU_CARD_HEIGHT:
                    selected[0] = mode

    cv2.namedWindow(title,cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(title,on_mouse)
    try:
        while selected[0] is None:
            cv2.imshow(title,image)
            key = cv2.waitKey(30)&0xFF
            if ord('1') <= key < ord('1')+len(EFFECTS):
                selected[0] = EFFECTS[key-ord('1')][0]
            elif key in (ord('q'),27):
                return None
            if cv2.getWindowProperty(title,cv2.WND_PROP_VISIBLE)<1:
                return None
        return selected[0]
    finally:
        cv2.destroyWindow(title)
