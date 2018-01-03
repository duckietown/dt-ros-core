from collections import OrderedDict

from anti_instagram import AntiInstagramInterface
from duckietown_segmaps.draw_map_on_images import plot_map
from duckietown_segmaps.maps import FRAME_AXLE, plot_map_and_segments, FRAME_GLOBAL
from duckietown_segmaps.transformations import TransformationsInfo
import duckietown_utils as dtu
from easy_algo import get_easy_algo_db
from easy_node.utils.timing import ProcessingTimingStats, FakeContext
from ground_projection import GroundProjection
from ground_projection.ground_projection_interface import find_ground_coordinates
from ground_projection.segment import rectify_segments
from lane_filter import FAMILY_LANE_FILTER
from lane_filter_generic import LaneFilterMoreGeneric
from line_detector.line_detector_interface import FAMILY_LINE_DETECTOR
from line_detector.visual_state_fancy_display import vs_fancy_display
from localization_templates import FAMILY_LOC_TEMPLATES
import numpy as np


@dtu.contract(gp=GroundProjection, ground_truth='SE2|None', image='array[HxWx3](uint8)')
def run_pipeline(image, gp, line_detector_name, image_prep_name, lane_filter_name, anti_instagram_name,
                 all_details=False, ground_truth=None,
                 actual_map=None):
    """
        Image: numpy (H,W,3) == BGR
        Returns a dictionary, res with the following fields:

            res['input_image']

        ground_truth = pose
    """

    dtu.check_isinstance(image, np.ndarray)
    

    gpg = gp.get_ground_projection_geometry()

    res = OrderedDict()
    res['Raw input image'] = image
    algo_db = get_easy_algo_db()
    line_detector = algo_db.create_instance(FAMILY_LINE_DETECTOR, line_detector_name)
    lane_filter = algo_db.create_instance(FAMILY_LANE_FILTER, lane_filter_name)
    image_prep = algo_db.create_instance('image_prep', image_prep_name)
    ai = algo_db.create_instance(AntiInstagramInterface.FAMILY, anti_instagram_name)

    pts = ProcessingTimingStats()
    pts.reset()
    pts.received_message()
    pts.decided_to_process()
    
    
    if all_details:
        segment_list = image_prep.process(FakeContext(), image, line_detector, transform = None)

        res['segments_on_image_input'] = vs_fancy_display(image_prep.image_cv, segment_list)
        res['segments_on_image_resized'] = vs_fancy_display(image_prep.image_resized, segment_list)
    
    
    with pts.phase('calculate AI transform'):
        ai.calculateTransform(image)

    with pts.phase('apply AI transform'):

        transformed = ai.applyTransform(image)

        if all_details:
            res['image_input_transformed'] = transformed

    with pts.phase('edge detection'):
        # note: do not apply transform twice!
        segment_list2 = image_prep.process(pts, image,
                                           line_detector, transform=ai.applyTransform)

        if all_details:
            
            res['resized and corrected'] = image_prep.image_corrected
            
    dtu.logger.debug('segment_list2: %s' % len(segment_list2.segments))
    
    if all_details:
        res['segments_on_image_input_transformed'] = \
            vs_fancy_display(image_prep.image_cv, segment_list2)

    if all_details:
        res['segments_on_image_input_transformed_resized'] = \
            vs_fancy_display(image_prep.image_resized, segment_list2)



    if all_details:
        grid = get_grid(image.shape[:2])
        res['grid'] = grid
        res['grid_remapped'] = gpg.rectify(grid)

    with pts.phase('rectify'):
        rectified0 = gpg.rectify(res['image_input'])
    
    rectified = ai.applyTransform(rectified0)

    if all_details:
        res['image_input_rect'] = rectified

#     res['difference between the two'] = res['image_input']*0.5 + res['image_input_rect']*0.5

    with pts.phase('rectify_segments'):
        segment_list2_rect = rectify_segments(gpg, segment_list2)
        
    res['segments rectified on image rectified'] = \
        vs_fancy_display(rectified, segment_list2_rect)

    # Project to ground
    with pts.phase('find_ground_coordinates'):
        sg = find_ground_coordinates(gpg, segment_list2_rect)

    lane_filter.initialize()
    if all_details:
        res['prior'] = lane_filter.get_plot_phi_d()

    with pts.phase('lane filter update'):
        _likelihood = lane_filter.update(sg)

    with pts.phase('lane filter plot'):
        res['likelihood'] = lane_filter.get_plot_phi_d(ground_truth=ground_truth)
    easy_algo_db = get_easy_algo_db()

    if isinstance(lane_filter, (LaneFilterMoreGeneric)):
        template_name = lane_filter.localization_template
    else:
        template_name = 'DT17_template_straight_straight'
        dtu.logger.debug('Using default template %r for visualization' % template_name)

    localization_template = \
        easy_algo_db.create_instance(FAMILY_LOC_TEMPLATES, template_name)

    with pts.phase('lane filter get_estimate()'):
        est = lane_filter.get_estimate()

    # Coordinates in TILE frame
    g = localization_template.pose_from_coords(est)
    tinfo = TransformationsInfo()
    tinfo.add_transformation(frame1=FRAME_GLOBAL, frame2=FRAME_AXLE, g=g)

    if actual_map is not None:
#         sm_axle = tinfo.transform_map_to_frame(actual_map, FRAME_AXLE)
        res['real'] = plot_map_and_segments(actual_map, tinfo, sg.segments, dpi=120,
                                             ground_truth=ground_truth)


    assumed = localization_template.get_map()
    res['model assumed for localization'] = plot_map_and_segments(assumed, tinfo, sg.segments, dpi=120,
                                           ground_truth=ground_truth)

    dtu.logger.debug('plot_map')
    assumed_axle = tinfo.transform_map_to_frame(assumed, FRAME_AXLE)


    res['map reprojected on image'] = plot_map(rectified, assumed_axle, gpg,
                                  do_ground=False, do_horizon=True, do_faces=False, do_faces_outline=True,
                                  do_segments=False)
#     res['blurred']= cv2.medianBlur(image, 11)
    stats = OrderedDict()
    stats['estimate'] = est


    dtu.logger.info(pts.get_stats())
    return res, stats

def get_grid(shape, L=32, col={0: (255,0,0), 1: (0,255,0)}):
    """ Creates a grid of given shape """
    H, W = shape
    res = np.zeros((H, W, 3), 'uint8')
    for i in range(H):
        for j in range(W):
            cx = int(i / L)
            cy = int(j / L)
            coli = (cx + cy) % 2
            res[i,j,:] = col[coli]
    return res
