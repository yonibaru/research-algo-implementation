 ## Algorithms Module
# This module contains the algorithms used in the project.
import numpy
from helper_functions import * 

"""
An implementation of the algorithms described in the paper:
"Star-Tracker Algorithm for Smartphones and Commercial Micro-Drones" by Revital Marbel,Boaz Ben-Moshe and Roi Yozevitch (2020). https://www.mdpi.com/1424-8220/20/4/1106#
Programmers: Yoni Baruch, Yevgeny Ivanov and Daniel Isakov
Date: 01-05-2025
"""

# --- Algorithm 1 Implementation ---

# ! This algorithm is extremely slow, on my M1 Macbook, it iterates over ~100,000 stars a second. There are 10000~ stars in the BSC, and the algorithm's complexity is O(n^3).
# ! Ultimately, it would take this algorithm 10000^3/100000 seconds to run, which is approximately 112.2 days. To test this function, you must know manually the stars in the image, and then run the function with a smaller catalog of stars (100-300) while insuring the original 3 stars are in the catalog subset.
def stars_identification_bf(detected_stars: List[dict],star_catalog: BSC,camera_scaling_factor:float=1) -> Tuple[dict, dict, dict] | None:
    """
    Implements Algorithm 1: Stars identification BF algorithm (Section 2.2).

    Picks one random triplet of pixels from the frame and compares its pixel distances against the angular distances of all possible catalog star triplets to find the catalog triplet yielding the minimum RMS error (using Eq. 1).

    Args:
        detected_stars: A list of detected star pixels (x, y) in the frame, each a dictionary with keys 'x' and 'y'.
        star_catalog: A catalog of bright stars (BSC).
        camera_scaling_factor: A scaling factor to convert pixel distances to angular distances as mentioned in Equation 3. This is not mentioned in the paper, but it is a mathematical necessity to convert the pixel distances to angular distances. The scaling factor is the ratio of the pixel distance to the angular distance, which is a constant for a given camera and setup. Without, the algorithm would not work, and would never converge to a feasible solution.
    Returns:
        The BSC star triplet (s_i, s_j, s_t) that yielded the minimum
        RMS error when compared to the chosen pixel triplet, or None if
        insufficient pixels/stars are available.
        
    Example 1: empty picture without any visible stars via "empty_image.png", scaling factor = 1/16.30.
    >>> stars_identification_bf(detect_stars("test_image.png"),get_star_catalog(),1/16.30)
    None
    
    Example 2: 3 visible stars in the frame via 'test_image.png', scaling factor = 1/16.30, stars are Zaniah, Porrima and Auva.
    >>> stars_identification_bf(detect_stars("test_image.png"),get_star_catalog(),1/16.30)
    ({'B': 'η', 'N': 'Zaniah', 'C': 'Vir', 'Dec': -0.666944, 'F': '15', 'HR': 4689, 'K': '9500', 'RA': 184.976667, 'V': '3.89'}, {'B': 'γ', 'N': 'Porrima', 'C': 'Vir', 'Dec': -1.449444, 'F': '29', 'HR': 4825, 'K': '7500', 'RA': 190.415, 'V': '3.65'}, {'B': 'δ', 'N': 'Auva', 'C': 'Vir', 'Dec': 3.3975, 'F': '43', 'HR': 4910, 'K': '3050', 'RA': 193.900833, 'V': '3.38'})
    """
    # if len(detected_stars) < 3:
    #     print("Error: Need at least 3 stars in the catalog.")
    #     return None

    # # 1. Pick *a* triplet of stars <p1, p2, p3> from Frame
    # p1 = detected_stars[0]
    # p2 = detected_stars[1]
    # p3 = detected_stars[2]

    # pixel_triplet = ((p1["x"],p1["y"]),(p2["x"],p2["y"]), (p3["x"],p3["y"]))

    # # 2. Calculate sorted distances for the chosen pixel triplet
    # dp1 = calculate_pixel_distance(pixel_triplet[0][0],pixel_triplet[0][1],pixel_triplet[1][0],pixel_triplet[1][1]) * camera_scaling_factor
    # dp2 = calculate_pixel_distance(pixel_triplet[0][0],pixel_triplet[0][1],pixel_triplet[2][0],pixel_triplet[2][1]) * camera_scaling_factor
    # dp3 = calculate_pixel_distance(pixel_triplet[1][0],pixel_triplet[1][1],pixel_triplet[2][0],pixel_triplet[2][1]) * camera_scaling_factor
    # sorted_pixel_distances = sorted([dp1, dp2, dp3])

    # min_rms = float('inf')
    # best_catalog_triplet = None
    # # 3. Iterate 'for every 3 stars <si, sj, st> in BSC'
    # for catalog_triplet in itertools.combinations(star_catalog, 3):
    #     s_i, s_j, s_t = catalog_triplet

    #     # 4. Calculate sorted angular distances for the catalog triplet
    #     dsi = calculate_angular_distance(s_i['RA'],s_i['Dec'],s_j['RA'],s_j['Dec'])
    #     dsj = calculate_angular_distance(s_i['RA'],s_i['Dec'],s_t['RA'],s_t['Dec'])
    #     dst = calculate_angular_distance(s_j['RA'],s_j['Dec'],s_t['RA'],s_t['Dec'])
    #     sorted_catalog_distances = sorted([dsi, dsj, dst])

    #     # 5. Calculate RMS using Equation 1
    #     try:
    #         current_rms = calculate_rms_error_eq1(
    #             pixel_distances=sorted_pixel_distances,
    #             angular_distances=sorted_catalog_distances,
    #         )
    #     except ValueError as e:
    #         print(f"Warning: RMS calculation error for catalog triplet {s_i[0],s_j[0],s_t[0]}: {e}")
    #         continue # Skip this triplet if distances don't match

    #     # 6. Update minimum RMS and best matching catalog triplet
    #     if current_rms < min_rms:
    #         min_rms = current_rms
    #         best_catalog_triplet = catalog_triplet
                
    # # 7. Return the catalog triplet corresponding to the minimum RMS
    # return best_catalog_triplet
    pass

# --- Algorithm 2 Implementation ---

def stars_identification_improved(detected_stars: List[dict], spht: SPHT, al_parameter: float, camera_scaling_factor: float = 1) -> List[Tuple[Tuple[dict, dict, dict], Tuple[dict, dict, dict]]]:
    """
    Implements Algorithm 2: Stars identification improved algorithm (Section 3.2).

    Algorithm 2 assigns each star triplet in the frame with its matching star triplet from the catalog. For each match, the algorithm also sets the confidence parameter (number between 0 and 1).

    Args:
        detected_stars: A list of detected star pixels (x, y) in the frame, each a dictionary with keys 'x' and 'y'.
        spht: The pre-computed Star Pattern Hash Table.
        al_parameter: The Accuracy Level parameter used when the SPHT was created. Used here for consistent key generation.
        camera_scaling_factor: A scaling factor to convert pixel distances to angular distances as mentioned in Equation 3.
    Returns:
        A matching between the detected stars and the catalog stars. More precisely:
        A list of tuples where each tuple contains:
        - Index 0 contains a tuple containing the three stars in the frame: (s_i, s_j, s_t), each a dictionary with keys 'x' and 'y'.
        - Index 1 contains a tuple containing the three matches from BSC: (s_i', s_j', s_t'), each a dictionary with keys 'RA', 'Dec', and 'HR'.
        
    Example:
    >>> # Setup test data
    >>> detected_stars = [{'x': 100, 'y': 100}, {'x': 150, 'y': 150}, {'x': 200, 'y': 100}]
    >>> # Create mock SPHT with a single pattern
    >>> mock_spht = {'1.234-5.678-9.012': [
    ...     ({'HR': 4689, 'N': 'Zaniah', 'RA': 184.976667, 'Dec': -0.666944},
    ...      {'HR': 4825, 'N': 'Porrima', 'RA': 190.415, 'Dec': -1.449444},
    ...      {'HR': 4910, 'N': 'Auva', 'RA': 193.900833, 'Dec': 3.3975})
    ... ]}
    >>> # Mock the create_spht_key function to always return our test key
    >>> def mock_create_key(*args, **kwargs): return '1.234-5.678-9.012'
    >>> import builtins
    >>> original_getattr = builtins.__getattribute__
    >>> def mock_getattr(obj, name):
    ...     if name == 'create_spht_key' and obj.__name__ == 'helper_functions':
    ...         return mock_create_key
    ...     return original_getattr(obj, name)
    >>> builtins.__getattribute__ = mock_getattr
    >>> # When run with the mock data, should return our expected match
    >>> result = stars_identification_improved(detected_stars, mock_spht, 0.1)
    >>> # Check if the expected structure is returned
    >>> len(result) >= 1 and len(result[0]) == 2 and len(result[0][0]) == 3 and len(result[0][1]) == 3
    True
    >>> # Restore original function
    >>> builtins.__getattribute__ = original_getattr
    """
    pass


# --- Algorithm 3 Implementation ---

def validation_algorithm_orientation(detected_stars: List[dict], orientation_matrix: numpy.ndarray, bsc_catalog: BSC) -> float:
    """
    Implements Algorithm 3: Validation Algorithm for the Reported Orientation.
    
    Algorithm 3 has 2 targets: (i) validates the reported orientation and (ii) improves the accuracy of the RTA orientation result. In order to have a validorientation (T0), at least two stars from the frame need to be matched to corresponding stars from the BSC.

    Args:
        detected_stars: A list of detected star pixels (x, y) in the frame, each a dictionary with keys 'x' and 'y'.
        orientation_matrix: The orientation matrix of the camera. (3x3 numpy array)
        bsc_catalog: The Bright Star Catalog.

    Returns:
        The estimated orientation error (weighted RMS of angular distances),
        or float('inf') if no valid pairs are found or other error.
        
    Example:
    >>> # Setup test data
    >>> import numpy as np
    >>> detected_stars = [{'x': 205, 'y': 30},{'x': 135, 'y': 88},{'x': 46, 'y': 48}]
    >>> # Create a mock identity orientation matrix
    >>> # Create a small mock catalog with just two stars
    >>> mock_catalog = [
    ...     {'RA': 184.976667, 'Dec': -0.666944, 'HR': 4689, 'N': 'Zaniah'},
    ...     {'RA': 190.415, 'Dec': -1.449444, 'HR': 4825, 'N': 'Porrima'},
    ...     {'RA': 193.900833, 'Dec': 3.3975, 'HR': 4910, 'N': 'Auva'}
    ... ]
    >>> orientation_matrix = calculate_orientation_matrix(detected_stars, mock_catalog)
    >>> # When validation is perfect, error should be very small
    >>> error = validation_algorithm_orientation(detected_stars, orientation_matrix, mock_catalog)
    >>> error < 0.01
    True
    """
    pass

# --- Algorithm 4 Implementation ---

def best_match_confidence_algorithm(detected_stars: List[dict],ism_data: List[Tuple[Tuple[dict, dict, dict], Tuple[dict, dict, dict]]],) -> List[Tuple[dict, dict, float]]:
    """
    Implements Algorithm 4: Best match confidence algorithm.

    Determines the best catalog star label for each pixel star and its confidence
    based on the frequency of matches found by Algorithm 2 (passed as ism_data).

    Args:
        detected_stars: A list of detected star pixels (x, y) in the frame, each a dictionary with keys 'x' and 'y'.
        ism_data: Intermediate Star Matches - the output from Algorithm 2.
                  A list of triplets and their candidate triplet matches: ( [(s1,s2,s3), (s4,s5,s6)...], ... )

    Returns:
        A matching between the detected stars and the catalog stars. More precisely:
        A list of tuples, each representing some match of a star to a BSC entry. Each tuple contains:
        - Index 0 contains a dictionary with keys 'x' and 'y'.
        - Index 1 contains a dictionary with keys 'RA', 'Dec', and 'HR'.
        - Index 2 contains the confidence parameter (number between 0 and 1).
        
    Example:
    >>> # Setup test data
    ... detected_stars = [
    ...    {'x': 205, 'y': 30},
    ...    {'x': 135, 'y': 88},
    ...    {'x': 46, 'y': 48}
        ]
        # Create a simple SPHT
        catalog_triplet = [
    ...     {'RA': 184.976667, 'Dec': -0.666944, 'HR': 4689, 'N': 'Zaniah'},
    ...     {'RA': 190.415, 'Dec': -1.449444, 'HR': 4825, 'N': 'Porrima'},
    ...     {'RA': 193.900833, 'Dec': 3.3975, 'HR': 4910, 'N': 'Auva'}
        ]
        ism_data = [(detected_stars[0], bsc_catalog[0])]
    >>> # Run the algorithm
    >>> result = best_match_confidence_algorithm(detected_stars, ism_data)
    >>> # Verify expected structure of result
    >>> len(result) == 3
    True
    """
    pass