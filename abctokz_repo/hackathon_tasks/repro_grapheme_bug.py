from abctokz.utils.unicode import grapheme_clusters

def test_grapheme_clusters_zwj_zwnj():
    # Base char (क U+0915) + Halant (् U+094D) + ZWJ (U+200D)
    # This is a common sequence in Devanagari to form a 'half-form'.
    text = "\u0915\u094d\u200d"
    
    clusters = grapheme_clusters(text)
    
    print(f"Input text (hex): {' '.join(hex(ord(c)) for c in text)}")
    print(f"Clusters: {clusters}")
    print(f"Number of clusters: {len(clusters)}")
    
    # Expected: All three characters should be in ONE cluster.
    if len(clusters) == 1:
        print("RESULT: SUCCESS (ZWJ correctly attached)")
    else:
        print("RESULT: FAILURE (ZWJ broke the cluster)")

if __name__ == "__main__":
    test_grapheme_clusters_zwj_zwnj()
