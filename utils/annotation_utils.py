from typing import List, Tuple
import numpy as np
import cv2
from . import get_center_of_bbox, get_bbox_dimensions

options = {"ball": 0, "goalkeepers": 1, "players": 2, "referees": 3, "stats": 4}

def ellipse(frame: np.ndarray, bbox: List[float], colour: Tuple[int, int, int], tracker_id: int=None):
    y2 = int(bbox[3])   
    x_center, _ = get_center_of_bbox(bbox)
    width, _ = get_bbox_dimensions(bbox)

    cv2.ellipse(frame, 
                center=(x_center, y2), 
                axes=(width,                
                      int(0.35*width)),     
                angle=0,                    
                startAngle=-45,             
                endAngle=235,    
                color=colour, 
                thickness=2)
    
    if tracker_id:
        rect_width = 40
        rect_height = 20
        rect_x1 = x_center - rect_width // 2
        rect_x2 = x_center + rect_width // 2
        rect_y1 = y2 - rect_height // 2 + 15       
        rect_y2 = y2 + rect_height // 2 + 15       

        cv2.rectangle(frame, 
                    pt1=(rect_x1, rect_y1),   
                    pt2=(rect_x2, rect_y2),  
                    color=colour,
                    thickness=cv2.FILLED)

        text = str(tracker_id)
        font_face = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2

        (text_width, text_height), _ = cv2.getTextSize(text, font_face, font_scale, thickness)
        text_x = rect_x1 + (rect_width - text_width) // 2
        text_y = rect_y1 + (rect_height + text_height) // 2

        cv2.putText(frame, 
                    text=text, 
                    org=(text_x, text_y),  
                    fontFace=font_face, 
                    fontScale=font_scale, 
                    color=(0, 0, 0), 
                    thickness=thickness)

    return frame
    
def triangle(frame: np.ndarray, bbox: List[float], colour: Tuple[int, int, int]):
    y = int(bbox[1])   
    x, _ = get_center_of_bbox(bbox)

    triangle_points = np.array([
        [x, y],                
        [x - 8, y - 15],      
        [x + 8, y - 15],      
    ])   

    cv2.drawContours(frame, contours=[triangle_points], contourIdx=0, color=colour, thickness=cv2.FILLED)   
    cv2.drawContours(frame, contours=[triangle_points], contourIdx=0, color=(0, 0, 0), thickness=2)   # border  

    return frame 

def ball_possession_box(frame_num: int, frame: np.ndarray, ball_possession: np.ndarray) -> np.ndarray:
    h, w = frame.shape[:2]
    # Place box below camera movement stats (assume stats at y=0..60)
    box_x1, box_y1 = 10, 70
    box_x2, box_y2 = min(350, w-10), min(150, h-10)

    overlay = frame.copy()
    # Draw filled box (no red border)
    cv2.rectangle(overlay, pt1=(box_x1, box_y1), pt2=(box_x2, box_y2), color=(255, 255, 255), thickness=cv2.FILLED)
    alpha = 0.4
    cv2.addWeighted(src1=overlay, alpha=alpha, src2=frame, beta=1-alpha, gamma=0, dst=frame)

    # Title: 'Possession' in bold (thicker font)
    cv2.putText(frame, text="Possession", org=(box_x1+10, box_y1+35), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 0, 0), thickness=4)

    ball_possession_till_frame = ball_possession[:frame_num+1]
    team_1_num_frames = ball_possession_till_frame[ball_possession_till_frame==1].shape[0]
    team_2_num_frames = ball_possession_till_frame[ball_possession_till_frame==2].shape[0]
    total = team_1_num_frames + team_2_num_frames

    if total > 0:
        team_1_possession = int(round(team_1_num_frames / total, 2) * 100)
        team_2_possession = int(round(team_2_num_frames / total, 2) * 100)
    else:
        team_1_possession = 0
        team_2_possession = 0

    cv2.putText(frame, text=f"Team 1: {team_1_possession}%", org=(box_x1+10, box_y1+65), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 0, 0), thickness=3)
    cv2.putText(frame, text=f"Team 2: {team_2_possession}%", org=(box_x1+10, box_y1+105), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 0, 0), thickness=3)

    return frame