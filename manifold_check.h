#ifndef MANIFOLD_CHECK_H
#define MANIFOLD_CHECK_H

#include <Eigen/Core>
#include <vector>
#include <utility>

struct manifold_result {
	bool edge_manifold = false;
	bool vertex_manifold = false;
	int euler_characteristic = 0;
	int non_manifold_vertices = 0;
	int non_manifold_edges = 0;

	// Indices of non-manifold vertices (in the reindexed mesh)
	std::vector<int> non_manifold_vertex_indices;
	// Non-manifold edges as pairs of vertex indices (in the reindexed mesh)
	std::vector<std::pair<int, int>> non_manifold_edge_list;

	bool is_manifold() const { return edge_manifold && vertex_manifold; }
};

manifold_result check_manifold(const Eigen::MatrixXi& F);

#endif
