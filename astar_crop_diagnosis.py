"""
AI-Powered Crop Disease Detection and Treatment Recommendation
================================================================
Core algorithm : A* Search  (Unit II)
Supporting idea: Production System (Unit I) - the knowledge base of
                  IF <symptoms> THEN <disease> rules acts as production rules.

--------------------------------------------------------------------
WHY A* INSTEAD OF PLAIN BFS
--------------------------------------------------------------------
BFS only returns the FIRST disease node it reaches - it has no notion
of "how good" a match is. When two diseases share overlapping symptoms
(e.g. both "Early Blight" and "Leaf Spot" cause yellow leaves), BFS
cannot rank them.

A* fixes this with:
    f(n) = g(n) + h(n)

    g(n) = steps taken so far (depth in the search - cost of the path)
    h(n) = number of MISMATCHED symptoms between candidate disease n
           and the symptoms the user actually reported
           (a disease with h(n)=0 is a perfect symptom match)

Because h(n) is a literal count of remaining mismatches (it never
"pretends" a worse-than-real match is better), it never overestimates
the true remaining cost -> the heuristic is admissible and A* explores
optimally rather than degrading into plain greedy search.

The search space:
    Start node  -> "root" (just the crop the user picked)
    Child nodes -> every known disease of that crop (production rules)
    Goal test   -> the candidate disease with the lowest f(n) once all
                   candidates have been generated (single-ply search
                   space here, since the KB is flat symptom-sets, but
                   the algorithm generalises to deeper diagnosis trees)
"""

import heapq
import itertools


# ---------------------------------------------------------------------
# 1. KNOWLEDGE BASE  (Production System - Unit I)
#    IF symptoms match THEN disease is a candidate
# ---------------------------------------------------------------------
KNOWLEDGE_BASE = {
    'Tomato': {
        'Early Blight': {
            'symptoms': {'yellow_leaves', 'brown_spots', 'dark_edges'},
            'severity': 'medium',
            'treatment': ["Remove and destroy affected leaves", "Apply a copper-based or chlorothalonil fungicide"],
            'prevention': ["Rotate crops away from tomato/potato beds each season", "Water at the base rather than overhead"],
        },
        'Leaf Spot': {
            'symptoms': {'yellow_leaves', 'brown_spots'},
            'severity': 'low',
            'treatment': ["Avoid overhead watering", "Apply neem oil or a recommended fungicide"],
            'prevention': ["Space plants for good airflow", "Remove plant debris after harvest"],
        },
        'Late Blight': {
            'symptoms': {'dark_edges', 'wilting', 'white_mold'},
            'severity': 'high',
            'treatment': ["Destroy infected plants immediately", "Apply a systemic fungicide (e.g. metalaxyl)"],
            'prevention': ["Avoid working in fields when foliage is wet", "Use certified disease-free transplants"],
        },
    },
    'Potato': {
        'Late Blight': {
            'symptoms': {'dark_edges', 'wilting', 'white_mold'},
            'severity': 'high',
            'treatment': ["Destroy infected foliage", "Apply fungicide preventively during humid weeks"],
            'prevention': ["Hill soil well over developing tubers", "Avoid overhead irrigation"],
        },
        'Early Blight': {
            'symptoms': {'yellow_leaves', 'brown_spots', 'dark_edges'},
            'severity': 'medium',
            'treatment': ["Remove lower infected leaves", "Ensure balanced soil nitrogen"],
            'prevention': ["Rotate with non-solanaceous crops", "Remove volunteer plants and weeds"],
        },
        'Black Scurf': {
            'symptoms': {'stunted_growth', 'dark_lesions_on_stem'},
            'severity': 'low',
            'treatment': ["Use certified seed potatoes", "Improve soil drainage"],
            'prevention': ["Plant into warm soil to speed emergence", "Clean equipment between fields"],
        },
    },
    'Chili': {
        'Anthracnose': {
            'symptoms': {'sunken_spots', 'dark_edges', 'fruit_rot'},
            'severity': 'high',
            'treatment': ["Remove and destroy infected fruit", "Apply a protectant fungicide before flowering"],
            'prevention': ["Avoid overhead irrigation near harvest", "Use disease-free certified seed"],
        },
        'Powdery Mildew': {
            'symptoms': {'white_powder_coating', 'yellow_leaves'},
            'severity': 'low',
            'treatment': ["Apply sulfur-based fungicide", "Improve airflow between plants"],
            'prevention': ["Avoid dense planting", "Choose resistant varieties where available"],
        },
    },
    'Brinjal': {
        'Fruit Rot': {
            'symptoms': {'sunken_spots', 'fruit_rot', 'wilting'},
            'severity': 'high',
            'treatment': ["Remove and destroy infected fruit promptly", "Apply recommended fungicide spray"],
            'prevention': ["Avoid fruit contact with wet soil", "Practice crop rotation"],
        },
        'Bacterial Wilt': {
            'symptoms': {'wilting', 'stunted_growth'},
            'severity': 'medium',
            'treatment': ["Remove and destroy wilted plants", "Improve field drainage"],
            'prevention': ["Use resistant rootstock where available", "Avoid waterlogged fields"],
        },
    },
    'Onion': {
        'Purple Blotch': {
            'symptoms': {'dark_edges', 'brown_spots', 'wilting'},
            'severity': 'medium',
            'treatment': ["Apply a protectant fungicide at first sign", "Avoid overhead irrigation"],
            'prevention': ["Rotate with non-allium crops", "Ensure good field drainage"],
        },
        'Downy Mildew': {
            'symptoms': {'yellow_leaves', 'white_mold'},
            'severity': 'medium',
            'treatment': ["Apply systemic fungicide early", "Remove and destroy infected plants"],
            'prevention': ["Avoid dense planting for airflow", "Use certified disease-free sets"],
        },
    },
    'Cabbage': {
        'Black Rot': {
            'symptoms': {'yellow_leaves', 'dark_lesions_on_stem', 'wilting'},
            'severity': 'high',
            'treatment': ["Remove and destroy infected plants", "Avoid working fields when wet"],
            'prevention': ["Use certified disease-free seed", "Rotate with non-brassica crops"],
        },
        'Clubroot': {
            'symptoms': {'stunted_growth', 'wilting'},
            'severity': 'medium',
            'treatment': ["Improve soil drainage", "Adjust soil pH toward neutral/alkaline"],
            'prevention': ["Rotate brassicas on a long cycle", "Disinfect tools between fields"],
        },
    },
    'Wheat': {
        'Rust': {
            'symptoms': {'orange_pustules', 'yellow_leaves', 'stunted_growth'},
            'severity': 'high',
            'treatment': ["Apply triazole-based fungicide", "Plant rust-resistant varieties"],
            'prevention': ["Scout fields weekly from tillering onward", "Destroy volunteer wheat between seasons"],
        },
        'Powdery Mildew': {
            'symptoms': {'white_powder_coating', 'yellow_leaves'},
            'severity': 'low',
            'treatment': ["Apply sulfur-based fungicide", "Avoid dense sowing"],
            'prevention': ["Choose resistant cultivars", "Avoid excess nitrogen"],
        },
    },
    'Rice': {
        'Blast': {
            'symptoms': {'brown_spots', 'dark_edges', 'white_mold'},
            'severity': 'high',
            'treatment': ["Apply tricyclazole fungicide", "Avoid excess nitrogen application"],
            'prevention': ["Maintain shallow standing water consistently", "Balance nitrogen with potassium/silica"],
        },
        'Bacterial Leaf Blight': {
            'symptoms': {'yellow_leaves', 'wilting', 'dark_lesions_on_stem'},
            'severity': 'medium',
            'treatment': ["Use resistant varieties", "Apply copper oxychloride spray"],
            'prevention': ["Avoid excess nitrogen fertilization", "Use clean, disease-free seed"],
        },
    },
}


# ---------------------------------------------------------------------
# 2. A* SEARCH
# ---------------------------------------------------------------------
class DiagnosisNode:
    """A node in the A* search space: one candidate disease."""

    def __init__(self, disease_name, disease_data, g_cost, user_symptoms):
        self.name = disease_name
        self.data = disease_data
        self.g = g_cost
        self.h = self._heuristic(user_symptoms)
        self.f = self.g + self.h

    def _heuristic(self, user_symptoms):
        """
        h(n) = number of mismatched symptoms (symmetric difference)
        between this disease's known symptom set and what the user
        reported. Lower is better; 0 = perfect match.
        """
        known = self.data["symptoms"]
        return len(known.symmetric_difference(user_symptoms))

    def matched_count(self, user_symptoms):
        return len(self.data["symptoms"] & user_symptoms)

    def confidence(self, user_symptoms):
        total = len(self.data["symptoms"])
        if total == 0:
            return 0.0
        return round(self.matched_count(user_symptoms) / total * 100, 1)

    def __repr__(self):
        return f"{self.name} (f={self.f}, g={self.g}, h={self.h})"


def a_star_diagnose(crop, user_symptoms, verbose=True):
    """
    Runs A* over the disease knowledge-base for a given crop.

    Returns:
        ranked_results: list of DiagnosisNode sorted by f(n) ascending
        trace: list of strings describing each step (for explainability)
    """
    if crop not in KNOWLEDGE_BASE:
        raise ValueError(f"No knowledge base entry for crop '{crop}'")

    user_symptoms = set(user_symptoms)
    frontier = []              # priority queue: (f, tie_breaker, node)
    trace = []
    counter = itertools.count()  # stable tie-breaking for equal f-values

    trace.append(f"START: crop='{crop}', reported symptoms={sorted(user_symptoms)}")

    # Generate every candidate disease as a child of the root (g=1 step)
    for disease_name, disease_data in KNOWLEDGE_BASE[crop].items():
        node = DiagnosisNode(disease_name, disease_data, g_cost=1,
                              user_symptoms=user_symptoms)
        heapq.heappush(frontier, (node.f, next(counter), node))
        trace.append(
            f"  EXPAND candidate '{disease_name}': "
            f"g={node.g}, h={node.h} (unmatched symptoms), f={node.f}"
        )

    # Pop nodes off the priority queue in f(n) order -> ranked diagnosis
    ranked_results = []
    while frontier:
        f, _, node = heapq.heappop(frontier)
        ranked_results.append(node)

    trace.append("GOAL SELECTION: lowest f(n) node(s) chosen as best match(es)")
    for i, node in enumerate(ranked_results, start=1):
        trace.append(
            f"  #{i}: {node.name} -> f={node.f}  "
            f"(confidence={node.confidence(user_symptoms)}%)"
        )

    if verbose:
        for line in trace:
            print(line)

    return ranked_results, trace


# ---------------------------------------------------------------------
# 3. PLAIN BFS  (kept only as a baseline to compare against, Unit II)
# ---------------------------------------------------------------------
def bfs_diagnose(crop, user_symptoms):
    """
    Naive baseline: returns the FIRST disease (in insertion order) that
    shares at least one symptom with the user's input. Demonstrates why
    A* is a meaningful upgrade - BFS cannot rank ambiguous matches.
    """
    user_symptoms = set(user_symptoms)
    for disease_name, disease_data in KNOWLEDGE_BASE[crop].items():
        if disease_data["symptoms"] & user_symptoms:
            return disease_name
    return None


# ---------------------------------------------------------------------
# 4. DEMO / EXAMPLE RUN
# ---------------------------------------------------------------------
def print_treatment(crop, node):
    print(f"\n>>> DIAGNOSIS: {node.name} ({crop})")
    print(f">>> Confidence: {node.confidence(user_symptoms)}%")
    print(">>> Recommended treatment:")
    for step in node.data["treatment"]:
        print(f"    - {step}")


if __name__ == "__main__":
    print("=" * 70)
    print("AI-POWERED CROP DISEASE DETECTION - A* SEARCH DEMO")
    print("=" * 70)

    crop = "Tomato"
    user_symptoms = {"yellow_leaves", "brown_spots", "dark_edges"}

    print(f"\nCrop selected   : {crop}")
    print(f"Symptoms entered: {sorted(user_symptoms)}\n")
    print("-" * 70)
    print("A* SEARCH TRACE")
    print("-" * 70)

    ranked, trace = a_star_diagnose(crop, user_symptoms)

    best = ranked[0]
    print_treatment(crop, best)

    print("\n" + "-" * 70)
    print("FULL RANKED RESULTS (for comparison / viva explanation)")
    print("-" * 70)
    for i, node in enumerate(ranked, start=1):
        print(f"{i}. {node.name:<20} f={node.f}  h={node.h}  "
              f"confidence={node.confidence(user_symptoms)}%")

    print("\n" + "-" * 70)
    print("BASELINE COMPARISON: plain BFS (first match only, no ranking)")
    print("-" * 70)
    bfs_result = bfs_diagnose(crop, user_symptoms)
    print(f"BFS would simply return: '{bfs_result}'")
    print("-> Notice BFS gives no confidence score and cannot distinguish")
    print("   between multiple diseases that share symptoms; A* can.")

    # ------------------------------------------------------------
    # Second example: an AMBIGUOUS case where A* clearly wins
    # ------------------------------------------------------------
    print("\n" + "=" * 70)
    print("SECOND EXAMPLE: ambiguous symptoms (Rice)")
    print("=" * 70)
    crop2 = "Rice"
    user_symptoms2 = {"brown_spots", "dark_edges"}
    print(f"\nCrop selected   : {crop2}")
    print(f"Symptoms entered: {sorted(user_symptoms2)}\n")

    ranked2, trace2 = a_star_diagnose(crop2, user_symptoms2, verbose=False)
    for i, node in enumerate(ranked2, start=1):
        print(f"{i}. {node.name:<25} f={node.f}  h={node.h}  "
              f"confidence={node.confidence(user_symptoms2)}%")
    print(f"\n>>> A* selects '{ranked2[0].name}' as the best match "
          f"(lowest f, highest confidence).")
