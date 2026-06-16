#include "manifold_check.h"
#include "manifold_check.h"
#include <igl/is_edge_manifold.h>
#include <igl/is_vertex_manifold.h>
#include <igl/euler_characteristic.h>

manifold_result check_manifold(const Eigen::MatrixXi& F)
{
	manifold_result result;

	if (F.rows() == 0)
		return result;

	result.edge_manifold = igl::is_edge_manifold(F);
	result.vertex_manifold = igl::is_vertex_manifold(F);
	result.euler_characteristic = igl::euler_characteristic(F);

	// Count non-manifold vertices and collect their indices
	{
		Eigen::VectorXi B_vm;
		igl::is_vertex_manifold(F, B_vm);
		result.non_manifold_vertices = 0;
		result.non_manifold_vertex_indices.clear();
		for (int i = 0; i < B_vm.size(); ++i) {
			if (!B_vm(i)) {
				result.non_manifold_vertices++;
				result.non_manifold_vertex_indices.push_back(i);
			}
		}
	}

	// Count non-manifold edges and collect them
	// NOTE: igl::is_edge_manifold sets BE(i) = true/1 when edge IS manifold,
	//       so non-manifold edges have BE(i) == 0 (false).
	{
		Eigen::MatrixXi BF_em, E_em;
		Eigen::VectorXi EMAP_em, BE_em;
		igl::is_edge_manifold(F, BF_em, E_em, EMAP_em, BE_em);
		result.non_manifold_edges = 0;
		result.non_manifold_edge_list.clear();
		for (int i = 0; i < BE_em.size(); ++i) {
			if (!BE_em(i)) {
				result.non_manifold_edges++;
				result.non_manifold_edge_list.emplace_back(E_em(i, 0), E_em(i, 1));
			}
		}
	}

	return result;
}
