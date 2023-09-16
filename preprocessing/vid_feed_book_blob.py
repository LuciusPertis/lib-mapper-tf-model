import cv2 as cv
import numpy as np

#import video

# Initialize variables for draw_quad routines
drawing = False
FOCUS_VERTEX = (-1,-1)


AUTOGEN_SHAPES_SPATIAL = True

#global for callback
#shapes=[[]]


from multiprocessing.pool import ThreadPool
from collections import deque

from common import clock, draw_str, StatValue

class PTasks:
    """ parrallel processing manager

        each task should have a return type of list as
        [<process-tag>, ... <return-data>]

        URGENT Tags
        - URGENT_NORMAL : simple pull/empty pending upto pending[0].ready() and then push all available
        - URGENT_PUSH_FIRST : dont work 
    """
    threadn = max(1, cv.getNumberOfCPUs() - 2)
    pool = ThreadPool(processes = threadn)
    pending = deque()
    task_tag = {}
    TASKS_COUNT = 0

    URGENT_NORMAL = threadn
    URGENT_FLAG = 2
    URGENT_PUSH_FIRST = 0
    URGENT_PULL_ONLY = 3

    task_admit = deque()
    task_return =  deque()
    task_return_max_store = 100

    def __init__(self):
        pass
    
    @staticmethod
    def _PULL(pull_count=threadn):
        while pull_count and len(PTasks.pending) > 0:
            pull_count -= 1

            if PTasks.pending[0].ready():
                return_args = PTasks.pending.popleft().get()
                PTasks.task_tag[return_args[0]] -= 1

                if len(PTasks.task_return) > PTasks.task_return_max_store:
                    PTasks.task_return.popleft()
                #PTasks.task_return.append(return_args)
            else:
                break
        
        PTasks._PUSH(makepull=0)


    @staticmethod
    def _PUSH(makepull=1):
        if makepull and not len(PTasks.pending) < PTasks.threadn:
            PTasks._PULL(makepull)

        while  len(PTasks.pending) < PTasks.threadn and len(PTasks.task_admit):
            task = PTasks.task_admit.popleft()
            task = PTasks.pool.apply_async(
                task[0], task[1])
            PTasks.pending.append(task)

    @staticmethod
    def _get_task_idx():
        PTasks.TASKS_COUNT += 1
        return PTasks.TASKS_COUNT

    def push_task(self, process_task, params, tag, urgent=URGENT_NORMAL):
        #if urgent:  PTasks.task_admit.appendleft((process_task, params, tag))
        #else:       PTasks.task_admit.append((process_task, params, tag))
        
        PTasks.task_admit.append((process_task, params, tag))
        
        if tag in PTasks.task_tag:  PTasks.task_tag[tag] += 1
        else:                       PTasks.task_tag[tag] = 0
        
        PTasks._PUSH(urgent)
    
    def done_tasks(self, tag):
        PTasks._PULL()
        return not (tag in PTasks.task_tag and PTasks.task_tag[tag] > 0)
    
    def force_toggleST(self):
        if PTasks.threadn == 1:
            PTasks.threadn = max(1, cv.getNumberOfCPUs() - 2)
        else:
            PTasks.threadn = 1



# drawing routines

def draw_quad(img: np.ndarray, shapes, alpha=0.5, phandle: PTasks=None):
    #latency = StatValue()
    #ts = clock()

    overlay = img.copy()

    def drawTask(vs=[]):
         #draw area
        if len(vs) > 2:
            cv.fillPoly(overlay, [vs], (100, 0, 100))

        #draw lines
        if len(vs) >  1:
            for i in range(-1, len(vs)-1):
                cv.line(overlay, vs[i], vs[i+1], (100, 200, 0), 2)
        
        #draw vertex circles
        for (x, y) in vs:
            cv.circle(overlay, (x, y), 10, (100, 200, 0), 4)
        
        return ['drawTask']

    for vertices in shapes:
        vertices = np.array(vertices).astype(int)
        
        if phandle is None:
            drawTask(vertices)
        else:
            phandle.push_task(drawTask, [vertices], 'drawTask')

    if phandle is not None:
        while(not phandle.done_tasks('drawTask')):pass
    
    
    #latency.update(clock() - ts)
    #print(f"latency        : {latency.value*1000} ms")

    return cv.addWeighted(overlay, alpha, img, 1- alpha, 0)

class OptFlow:
    """
    The OptFlow class encapsulates optical flow calculations and visualization.

    This class is used to calculate and visualize optical flow between frames in a video sequence.
    Optical flow represents the motion of objects between consecutive frames in a video and can be
    used for various computer vision tasks, including object tracking and motion analysis.

    Usage in the Code:
    -----------------
    In the provided code, the OptFlow class is utilized to calculate and visualize optical flow
    between frames in a video stream. It plays a key role in spatially and temporally tracking
    objects within the video frames. Specifically, it is used to estimate the motion of objects
    and visualize the flow of pixels between frames. The calculated optical flow is displayed
    on the 'flow' window in the graphical user interface.

    Attributes:
    -----------
    - prev_gray: The grayscale representation of the previous frame.
    - show_hsv: A flag to determine whether to display optical flow in HSV color space.
    - show_flow: A flag to enable or disable the visualization of optical flow.
    - use_spatial_propagation: A flag to enable spatial propagation in the DIS optical flow algorithm.
    - use_temporal_propagation: A flag to enable temporal propagation in the DIS optical flow algorithm.
    - flow: The optical flow data representing motion between frames.
    
    Methods:
    --------
    - __init__(self, prev): Initializes the OptFlow object with the previous frame.
    - draw_flow(img, flow, step=48): Draws optical flow vectors on an image.
    - draw_hsv(flow): Converts optical flow data to an HSV color representation.
    - _warp_flow(img, flow): Warps an image based on optical flow.
    - get_spatial_displacement(self, x, y, radius=8, step=2): Calculates spatial displacement at a point.
    - calculate_frame_displacement(self, curr_frame, prev_frame=None): Calculates optical flow between frames.

    See Also:
    ---------
    - The OptFlow class is used in conjunction with the main code to visualize and analyze
      the motion of objects within video frames.

    """
    def __init__(self, prev):
        """
        Initializes the Optical Flow object.

        Args:
        - prev: The previous frame (numpy.ndarray) for optical flow calculation.
        """
        cv.namedWindow('flow', cv.WINDOW_NORMAL)
        self.prev_gray = cv.cvtColor(prev, cv.COLOR_BGR2GRAY)
        self.show_hsv = False
        self.show_flow = True
        self.use_spatial_propagation = True
        self.use_temporal_propagation = True
        self.flow = None

        self.inst = cv.DISOpticalFlow.create(cv.DISOPTICAL_FLOW_PRESET_MEDIUM)
        self.inst.setUseSpatialPropagation(self.use_spatial_propagation)
    
    @staticmethod
    def draw_flow(img, flow, step=48):
        """
        Draws optical flow vectors on an image.

        Args:
        - img: The input image (numpy.ndarray).
        - flow: Optical flow data (numpy.ndarray).
        - step: Step size for drawing flow vectors.

        Returns:
        - vis: An image with optical flow vectors drawn (numpy.ndarray).
        """
        h, w = img.shape[:2]
        y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2,-1).astype(int)
        fx, fy = flow[y,x].T
        lines = np.vstack([x, y, x+fx, y+fy]).T.reshape(-1, 2, 2)
        lines = np.int32(lines + 0.5)
        vis = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
        cv.polylines(vis, lines, 0, (0, 255, 0), step//16)
        for (x1, y1), (_x2, _y2) in lines:
            cv.circle(vis, (x1, y1), step//10, (0, 255, 0), -1)
        return vis

    @staticmethod
    def draw_hsv(flow):
        h, w = flow.shape[:2]
        fx, fy = flow[:,:,0], flow[:,:,1]
        ang = np.arctan2(fy, fx) + np.pi
        v = np.sqrt(fx*fx+fy*fy)
        hsv = np.zeros((h, w, 3), np.uint8)
        hsv[...,0] = ang*(180/np.pi/2)
        hsv[...,1] = 255
        hsv[...,2] = np.minimum(v*4, 255)
        bgr = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
        return bgr

    @staticmethod    
    def _warp_flow(img, flow):
        """
        Warps an image based on optical flow.

        Args:
        - img: The input image (numpy.ndarray).
        - flow: Optical flow data (numpy.ndarray).

        Returns:
        - res: Warped image (numpy.ndarray).
        """
        h, w = flow.shape[:2]
        flow = -flow
        flow[:,:,0] += np.arange(w)
        flow[:,:,1] += np.arange(h)[:,np.newaxis]
        res = cv.remap(img, flow, None, cv.INTER_LINEAR)
        return res

    def get_spatial_displacement(self, x: int, y: int, radius=8, step=2):
        """
        Calculates the spatial displacement at a specific point using optical flow.

        Args:
        - x: X-coordinate of the point.
        - y: Y-coordinate of the point.
        - radius: Radius for sampling nearby flow vectors.
        - step: Step size for sampling.

        Returns:
        - fx_mean: Mean horizontal displacement.
        - fy_mean: Mean vertical displacement.
        """
        x, y = int(x), int(y)
        h, w = self.prev_gray.shape[:2]
        fx, fy = self.flow[max(0, y-radius):min(h, y+radius):step , max(0, x-radius):min(w, x+radius):step].T.reshape(2, -1)

        return fx.mean(), fy.mean()

    def calculate_frame_displacement(self, curr_frame, prev_frame=None):
        """
        Calculates optical flow between two frames.

        Args:
        - curr_frame: The current frame (numpy.ndarray).
        - prev_frame: The previous frame (numpy.ndarray).

        Returns:
        None
        """
        # Convert frames to grayscale for feature matching
        if prev_frame is not None:
            self.prev_gray = cv.cvtColor(prev_frame, cv.COLOR_BGR2GRAY)
        curr_gray = cv.cvtColor(curr_frame, cv.COLOR_BGR2GRAY)

        if self.flow is not None and self.use_temporal_propagation:
            #warp previous flow to get an initial approximation for the current flow:
            self.flow = self.inst.calc(self.prev_gray, curr_gray, OptFlow._warp_flow(self.flow, self.flow))
        else:
            self.flow = self.inst.calc(self.prev_gray, curr_gray, None)
        
        self.prev_gray = curr_gray
        if self.show_flow:
            
            cv.imshow('flow', OptFlow.draw_flow(self.prev_gray, self.flow))
    



# Define mouse callback function
def clicked_on_vertex(shapes, x, y, radius, thickness):
    for s_index in range(len(shapes)):
        for v_index in range(len(shapes[s_index])):
            if  abs(x-shapes[s_index][v_index][0]) < radius + thickness and \
                abs(y-shapes[s_index][v_index][1]) < radius + thickness:
                return s_index, v_index
    return -1, -1

def make_quad_mouse_cb(event, x, y, flags, param):
    global drawing, FOCUS_VERTEX

    shapes = param[0]
    vertices = shapes[FOCUS_VERTEX[0]]

    if event == cv.EVENT_LBUTTONDOWN:
        si, vi = clicked_on_vertex(shapes, x,y, 10,4)
        
        if si != -1:
            #change previously defined vextex
            shapes[si][vi]=[x,y]
            FOCUS_VERTEX = (si,vi)
            drawing = True

        elif len(vertices) < 4:
            vertices.append([x, y])
        else:
            shapes.append([[x,y]])
        
    elif event == cv.EVENT_MOUSEMOVE and drawing:
        shapes[FOCUS_VERTEX[0]][FOCUS_VERTEX[1]] = [x, y]

    elif event == cv.EVENT_LBUTTONUP:
        FOCUS_VERTEX = (-1, -1)
        drawing = False
    
    elif event == cv.EVENT_RBUTTONUP:
        FOCUS_VERTEX = (-1, -1)
        if len(shapes[-1]):
            shapes[-1].pop()
        elif len(shapes)>1:
            shapes.pop()
        drawing = False
    
    #print('LOG: (len:', len(shapes),')', shapes[-1], 'pointer: ', (x,y), "drawing :", drawing)


# Helper functions

def save_buffer_to_file(buff, path_to_file):
    if type(buff) == type(""):
        with open(path_to_file, 'a') as annof:
            annof.write(buff)
        
    elif type(buff) == type([[]]):
        with open(path_to_file, 'a') as annof:
            for ms in buff:
                #annof.write(f'\n{FRAME_COUNTER}_{len(shapes)}')
                annof.write(f'\n{ms[0]}_{len(ms[1])}')
                for s in ms[1]:
                    annof.write(f'\n{len(s)}')
                    for v in s:
                        annof.write(f' {v[0],v[1]}')

if __name__ == "__main__":
    # Open video file
    rpath, video_file, ext = 'ModelData/vids/', 'VID_20230824_143549', '.mp4' 
    cap = cv.VideoCapture(rpath+video_file+ext)
    cv.namedWindow('Video Frame', cv.WINDOW_NORMAL)


    # Initialize the previous frame as the first frame of the video
    prev_frame = cap.read()[1]
    H_, W_ = prev_frame.shape[:2]
    FRAME_COUNTER = 0
    FRAME_SKIP_STRIDE = 5

    # Create an instance of the OptFlow class for optical flow calculations
    optdis = OptFlow(prev_frame)
    # flag to control optical flow calculations refresh
    refresh_opt = True

    # Buffers
    BUFFER = ""
    SHAPE_BUFFER = []
    BUFFER_SELECT = 0
    shapes = [[]]
    mouse_work = 0

    # Create a parallel tasks manager for efficient processing
    ptasks = PTasks()
    #ptasks.force_toggleST()

    # Initialize a flag for testing purposes
    _rtest = True

    # start Video Ingestion
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        #optdis.calculate_frame_displacement(frame, prev_frame)
        #dx, dy = optdis.get_spatial_displacement(W_//2, H_//2, radius=100, step=5)
        #print(f"Displacement [frame {FRAME_COUNTER}]: ({dx}, {dy})")
        
        latency = StatValue()
        ts = clock()
        if AUTOGEN_SHAPES_SPATIAL and len(shapes[0]):
            # Calculate optical flow displacement for shape tracking
            optdis.calculate_frame_displacement(frame, [None, prev_frame][refresh_opt])
            print("FRAME_AVG_DIS: ", optdis.get_spatial_displacement(W_//2, H_//2), "refresh_opt:", refresh_opt)
            refresh_opt = False

            # Parallel task for spatial displacement calculation
            def task_shape_camshift(si: int, vi: int):
                #print('shapes:', shapes)
                try:
                    dx, dy = optdis.get_spatial_displacement(shapes[si][vi][0], shapes[si][vi][1], radius=1, step=1)
                    #print(dx, dy, shapes[si][vi], end=' ')
                    shapes[si][vi][0] = max(0 , min(W_ , shapes[si][vi][0 ]+dx))
                    shapes[si][vi][1] = max(0 , min(H_ , shapes[si][vi][1]+dy))
                    #print(shapes[si][vi])
                except Exception as e:
                    print(e)
                    pass
                finally:
                    return ['camshift']
            
            # Push spatial displacement tasks to the parallel task manager
            for i in range(len(shapes)):
                for j in range(len(shapes[i])):
                    if _rtest:
                        ptasks.push_task(task_shape_camshift, (i, j), 'camshift', PTasks.URGENT_NORMAL)
                    else:
                        task_shape_camshift(i,j)

        # Updated callback
        #TODO: see if not required and can be called outside [while cap.isOpened()]
        cv.setMouseCallback('Video Frame', make_quad_mouse_cb, [shapes, mouse_work])

        # wait for shape_auto_camshift
        while(not ptasks.done_tasks('camshift')): 
            pass

        latency.update(clock() - ts)
        print(f"latency        : {latency.value*1000} ms")

        # MAIN loop to draw quads on single frame and waitKey(actions)
        key = 0
        while key != 27:
            # Check if the current frame should be skipped
            if FRAME_COUNTER % FRAME_SKIP_STRIDE != 0:
                break

            # Display the frame with the quadrilaterals
            frame_copy = draw_quad(frame, shapes) # , phandle=ptasks) :: not efficient
            # Display the frame with quadrilaterals
            cv.imshow('Video Frame', frame_copy) 

            # Check for key events
            key = cv.waitKey(2) & 0xFF

            # Key actions 
            if   key == ord('n') or key == ord(' '):
                # Save the frame annotations in SHAPE_BUFFER
                if len(shapes[0]):
                    filename = f'{video_file}'
                    annotations = {'>> frame_no': FRAME_COUNTER, 'filename': filename, 'shapes': len(shapes)}
                    
                    if BUFFER_SELECT:       
                        # Save annotations to a string buffer
                        BUFFER += f'\n{cap.get(cv.CV_CAP_PROP_POS_FRAMES)-1}_{len(shapes)}'
                        for s in shapes:
                            BUFFER += f'\n{len(s)}'
                            for v in s:
                                BUFFER += f' {v[0],v[1]}'
                    else:
                        # Store annotations in an Obj
                        SHAPE_BUFFER.append([cap.get(cv.CV_CAP_PROP_POS_FRAMES)-1, shapes])

                    print(annotations)

                break # next frame
            elif key == ord('l'):
                # mouse_work
                pass
            elif key == ord('x'):
                print(">> Clearing shapes")
                shapes = [[]]
                refresh_opt = True
                break
            elif key == ord('b'):
                BUFFER_SELECT = (BUFFER_SELECT + 1)%2
                print(">> BUFFER_SELECT: ", ["text", "shape"][BUFFER_SELECT])
            elif key == ord('r'):
                _rtest = not _rtest
                print(">> _rtest:", _rtest)
            elif key == ord('-'):
                FRAME_SKIP_STRIDE = max(1, FRAME_SKIP_STRIDE-1)
                print(">> decreasing frame_stride: ", FRAME_SKIP_STRIDE)
            elif key == ord('+'):
                FRAME_SKIP_STRIDE = FRAME_SKIP_STRIDE+1
                print(">> increasing frame_stride: ", FRAME_SKIP_STRIDE)
        # while key != 27: END 

        FRAME_COUNTER += 1
        prev_frame = frame.copy()

        if key == 27:
            break
    # while cap.isOpened(): END

    # Save shape annotations to a file
    save_buffer_to_file(SHAPE_BUFFER, rpath+f'{video_file}_frameAnno.txt')

    # Release the video capture object and close all OpenCV windows
    cap.release()
    cv.destroyAllWindows()


