import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score

from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist, squareform

# -----------------------------------------------------------------
# STEP 1: Sample document corpus (3 natural topic groups)
# -----------------------------------------------------------------
documents = [
    # Sports
    "The cricket team won the match after the batsman scored a great century.",
    "The football team played well and the coach praised the team after the match win.",
    "The tennis player won the match and secured the championship title after a tough game.",
    "The basketball team won the match after the player scored the winning point.",
    # Technology
    "The new smartphone has a fast processor and the camera software is very good.",
    "Artificial intelligence and machine learning software are transforming the computer industry.",
    "The laptop has a powerful processor and GPU for machine learning software tasks.",
    "Cloud computing software helps the company scale computer applications for the industry.",
    # Health
    "Doctors recommend regular exercise and a balanced diet for good health of the body.",
    "The hospital doctors introduced a new health treatment to reduce disease and improve health.",
    "Drinking enough water and regular exercise improves body health and overall energy.",
    "Doctors say meditation, exercise and proper sleep improve health and reduce stress.",
]

labels_true = (["Sports"] * 4) + (["Technology"] * 4) + (["Health"] * 4)

print("=" * 70)
print("STEP 1: Corpus")
print("=" * 70)
for i, d in enumerate(documents):
    print(f"D{i+1:02d} [{labels_true[i]:10s}]: {d}")

# -----------------------------------------------------------------
# STEP 2: Text vectorization using TF-IDF
# -----------------------------------------------------------------
vectorizer = TfidfVectorizer(stop_words="english")
tfidf_matrix = vectorizer.fit_transform(documents)
tfidf_dense = tfidf_matrix.toarray()

print("\n" + "=" * 70)
print("STEP 2: TF-IDF matrix shape ->", tfidf_dense.shape)
print("=" * 70)
print("Vocabulary sample:", list(vectorizer.vocabulary_.keys())[:10])

# -----------------------------------------------------------------
# STEP 3: Compute pairwise distance matrix (cosine distance)
# -----------------------------------------------------------------
dist_matrix = pdist(tfidf_dense, metric="cosine")
square_dist = squareform(dist_matrix)

print("\n" + "=" * 70)
print("STEP 3: Pairwise cosine distance matrix (rounded)")
print("=" * 70)
np.set_printoptions(precision=2, suppress=True)
print(square_dist)

# -----------------------------------------------------------------
# STEP 4: Agglomerative clustering using Scipy linkage (for dendrogram)
# -----------------------------------------------------------------
Z = linkage(dist_matrix, method="average")  # average linkage

print("\n" + "=" * 70)
print("STEP 4: Linkage matrix (scipy)")
print("=" * 70)
print("[idx1, idx2, distance, sample_count]")
print(np.round(Z, 3))

doc_labels = [f"D{i+1}\n({labels_true[i]})" for i in range(len(documents))]

plt.figure(figsize=(11, 6))
dendrogram(
    Z,
    labels=doc_labels,
    leaf_font_size=9,
    color_threshold=0.7,
)
plt.title("Dendrogram - Agglomerative Hierarchical Clustering\n(Average Linkage, Cosine Distance)")
plt.xlabel("Documents")
plt.ylabel("Cosine Distance")
plt.axhline(y=0.7, color="gray", linestyle="--", linewidth=1)
plt.tight_layout()
plt.savefig("/home/claude/practical/dendrogram.png", dpi=150)
plt.close()
print("\nDendrogram saved -> dendrogram.png")

# -----------------------------------------------------------------
# STEP 5: Cut the dendrogram / fit AgglomerativeClustering (k = 3)
# -----------------------------------------------------------------
k = 3
model = AgglomerativeClustering(n_clusters=k, metric="cosine", linkage="average")
cluster_ids = model.fit_predict(tfidf_dense)

print("\n" + "=" * 70)
print(f"STEP 5: Agglomerative Clustering result (k = {k})")
print("=" * 70)
for i, (doc, cid) in enumerate(zip(documents, cluster_ids)):
    print(f"D{i+1:02d} -> Cluster {cid}  | True topic: {labels_true[i]:10s} | {doc[:55]}...")

sil = silhouette_score(tfidf_dense, cluster_ids, metric="cosine")
print(f"\nSilhouette Score (cosine): {sil:.4f}")

# -----------------------------------------------------------------
# STEP 6: Cluster -> Top keywords per cluster (interpretation)
# -----------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 6: Top TF-IDF keywords per cluster")
print("=" * 70)
terms = np.array(vectorizer.get_feature_names_out())
for c in range(k):
    idxs = np.where(cluster_ids == c)[0]
    cluster_tfidf = tfidf_dense[idxs].mean(axis=0)
    top_idx = cluster_tfidf.argsort()[::-1][:6]
    print(f"Cluster {c}: {', '.join(terms[top_idx])}")

# -----------------------------------------------------------------
# STEP 7: Divisive clustering (conceptual demo) via repeated bisection
#          scikit-learn has no built-in divisive clustering, so we
#          demonstrate the idea using recursive KMeans(k=2) bisection.
# -----------------------------------------------------------------
from sklearn.cluster import KMeans

def divisive_bisect(X, doc_idxs, depth, max_depth, results):
    if depth == max_depth or len(doc_idxs) <= 1:
        results.append(doc_idxs)
        return
    km = KMeans(n_clusters=2, n_init=10, random_state=42)
    sub = X[doc_idxs]
    lbl = km.fit_predict(sub)
    left = [doc_idxs[i] for i in range(len(doc_idxs)) if lbl[i] == 0]
    right = [doc_idxs[i] for i in range(len(doc_idxs)) if lbl[i] == 1]
    divisive_bisect(X, left, depth + 1, max_depth, results)
    divisive_bisect(X, right, depth + 1, max_depth, results)

results = []
all_idxs = list(range(len(documents)))
divisive_bisect(tfidf_dense, all_idxs, depth=0, max_depth=2, results=results)

print("\n" + "=" * 70)
print("STEP 7: Divisive Clustering demo (recursive bisecting K-Means)")
print("=" * 70)
for cnum, group in enumerate(results):
    docs_in_group = [f"D{i+1}({labels_true[i]})" for i in group]
    print(f"Divisive Cluster {cnum}: {docs_in_group}")

print("\nDone.")
