#ifndef __MAPPING_H__
#define __MAPPING_H__

#include <vector>
#include "pointset.h"
#include "polygon.h"

//---------------------------------------------------------------------------------------//
//---------- 							D E F I N E S 							---------//
//---------------------------------------------------------------------------------------//

//---------------------------------------------------------------------------------------//
//---------					  D A T A   T Y P E   D E C L S				        ---------//
//---------------------------------------------------------------------------------------//


//---------------------------------------------------------------------------------------//
//	STRUCTURE		:	[similarity_matrix_entry]
//  MADE BY			:	[JSC] Jefferson Salcedo Chavez		DATE: FEB/2026
//	DESCRIPTION		:	Entrada de la matriz de similitud entre dos polígonos
//                      Almacena índices, métricas y geometría de intersección
//---------------------------------------------------------------------------------------//
struct similarity_matrix_entry
{
	const polygon_reference* poly_l1_ref;  // Referencia al polígono de L1
	const polygon_reference* poly_l2_ref;  // Referencia al polígono de L2

	// Métricas
	double similarity_index;     // Índice de similitud (Jaccard modificado)
	double intersection_area;    // Área de intersección
	double area_l1;              // Área del polígono L1
	double area_l2;              // Área del polígono L2

	// Geometría de la intersección
	Eigen::MatrixXd vertices;    // Vértices de la intersección
	Eigen::MatrixXi faces;       // Caras de la intersección
};

//---------------------------------------------------------------------------------------//
//	STRUCTURE		:	[mapping_group]
//  MADE BY			:	[JSC] Jefferson Salcedo Chavez		DATE: FEB/2026
//	DESCRIPTION		:	Grupo de correspondencia entre polígonos de L1 y L2
//                      Resultado del algoritmo BFS de agrupación
//---------------------------------------------------------------------------------------//
struct mapping_group
{
	std::vector<int> polygons_l1;  // IDs de polígonos de L1 en el grupo
	std::vector<int> polygons_l2;  // IDs de polígonos de L2 en el grupo
};

//---------------------------------------------------------------------------------------//
//	FUNCION NAME	:	[compute_similarity_matrix]
//  MADE BY:			[JSC] Jefferson Salcedo Chavez				DATE: FEB/2026
//	DESCRIPTION		:	Computes the similarity matrix between L1 and L2 polygons
//---------------------------------------------------------------------------------------//
std::vector<similarity_matrix_entry>
	compute_similarity_matrix
	(
		const std::vector<polygon_reference>&	refs_l1,		// < L1 polygon references
		const std::vector<polygon_reference>&	refs_l2,		// < L2 polygon references
		const std::vector<solidpolygon>&		all_polygons,	// < All polygons array
		const std::vector<datapoints>&			contours_l1,	// < L1 contours data
		const std::vector<datapoints>&			contours_l2		// < L2 contours data
	);

//---------------------------------------------------------------------------------------//
//	FUNCION NAME	:	[compute_mapping_groups]
//  MADE BY:			[JSC] Jefferson Salcedo Chavez				DATE: FEB/2026
//	DESCRIPTION		:	Computes the mapping groups between L1 and L2 polygons using BFS
//---------------------------------------------------------------------------------------//
std::vector<mapping_group>
	compute_mapping_groups
	(
		const std::vector<solidpolygon>&		polygons_l1,	// < L1 solid polygons
		const std::vector<datapoints>&		contours_l1,	// < L1 contours data
		const std::vector<solidpolygon>&		polygons_l2,	// < L2 solid polygons
		const std::vector<datapoints>&		contours_l2,	// < L2 contours data
		double								threshold		// < Similarity threshold
	);

#endif 