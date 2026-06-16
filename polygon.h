#ifndef __POLYGON_H__
#define __POLYGON_H__

#include <vector>
#include <string>
#include <Eigen/Core>
#include "pointset.h"
#include "tree.h"

//---------------------------------------------------------------------------------------//
//---------- 							D E F I N E S 							---------//
//---------------------------------------------------------------------------------------//

//---------------------------------------------------------------------------------------//
//---------					  D A T A   T Y P E   D E C L S				        ---------//
//---------------------------------------------------------------------------------------//

//---------------------------------------------------------------------------------------//
//	STRUCTURE		:	[solidpolygon]
//  MADE BY			:	[JSC] Jefferson Salcedo Chavez		DATE: FEB/2026
//	DESCRIPTION		:	Polígono sólido definido por contornos (exterior + huecos)
//                      Representa la geometría del polígono mediante índices
//---------------------------------------------------------------------------------------//
struct solidpolygon
{
	std::vector<int> contours;  // Primer elemento: contorno exterior
	                             // Elementos restantes: contornos de huecos
};

//---------------------------------------------------------------------------------------//
//	STRUCTURE		:	[polygon_reference]
//  MADE BY			:	[JSC] Jefferson Salcedo Chavez		DATE: FEB/2026
//	DESCRIPTION		:	Referencia ligera a un solidpolygon en el arreglo global
//                      Almacena índice en all_polygons y nivel (L1/L2)
//---------------------------------------------------------------------------------------//
struct polygon_reference
{
	int polygon_id;  // Índice en el arreglo global all_polygons
	int level;       // Nivel: 0 = L1 (nivel inferior), 1 = L2 (nivel superior)
	
	// Constructores
	polygon_reference(int id, int lvl) : polygon_id(id), level(lvl) {}
	polygon_reference() : polygon_id(-1), level(-1) {}
};

//---------------------------------------------------------------------------------------//
//	STRUCTURE		:	[polygon_pair]
//  MADE BY			:	[JSC] Jefferson Salcedo Chavez		DATE: FEB/2026
//	DESCRIPTION		:	Par de polígonos para algoritmos BFS (mapping groups)
//                      Usa polygon_reference para identificación
//---------------------------------------------------------------------------------------//
struct polygon_pair
{
	polygon_reference ref;  // Referencia al polígono (ID + nivel)
	
	// Constructor
	polygon_pair(const polygon_reference& r) : ref(r) {}
	polygon_pair() : ref() {}
};

//---------------------------------------------------------------------------------------//
//	STRUCTURE		:	[polygon_intersection_result]
//  MADE BY			:	[JSC] Jefferson Salcedo Chavez		DATE: FEB/2026
//	DESCRIPTION		:	Resultado del cálculo de intersección entre dos polígonos
//                      Incluye área y geometría (vértices y caras)
//---------------------------------------------------------------------------------------//
struct polygon_intersection_result
{
	double intersection_area;    // Área de la región de intersección
	
	// Geometría de la intersección (para visualización)
	Eigen::MatrixXd vertices;    // Matriz Nx3 de vértices
	Eigen::MatrixXi faces;       // Matriz Mx3 de triángulos (índices)
};

//---------------------------------------------------------------------------------------//
//	FUNCION NAME	:	[pointInPolygon]
//  MADE BY:			[JSC] Jefferson Salcedo Chavez				DATE: FEB/2026
//	DESCRIPTION		:	Checks if a point is inside a polygon using ray casting
//---------------------------------------------------------------------------------------//
bool
	pointInPolygon
	(
		const Eigen::Vector3d&					p,		// < Point to test
		const std::vector<Eigen::Vector3d>&		poly	// < Polygon vertices
	);

//---------------------------------------------------------------------------------------//
//	FUNCION NAME	:	[polygonArea]
//  MADE BY:			[JSC] Jefferson Salcedo Chavez				DATE: FEB/2026
//	DESCRIPTION		:	Calculates the area of a polygon using the shoelace formula
//---------------------------------------------------------------------------------------//
double
	polygonArea
	(
		const std::vector<Eigen::Vector3d>&		poly	// < Polygon vertices
	);

//---------------------------------------------------------------------------------------//
//	FUNCION NAME	:	[point_in_polygon_2d_geogram]
//  MADE BY:			[JSC] Jefferson Salcedo Chavez				DATE: FEB/2026
//	DESCRIPTION		:	Checks if a 2D point is inside a polygon (geogram variant)
//---------------------------------------------------------------------------------------//
bool
	point_in_polygon_2d_geogram
	(
		const Eigen::Vector2d&					p,		// < 2D point to test
		const std::vector<Eigen::Vector3d>&		poly	// < Polygon vertices
	);

//---------------------------------------------------------------------------------------//
//	FUNCION NAME	:	[compute_polygon_area]
//  MADE BY:			[JSC] Jefferson Salcedo Chavez				DATE: FEB/2026
//	DESCRIPTION		:	Computes the area of a solid polygon (exterior minus holes)
//---------------------------------------------------------------------------------------//
double
	compute_polygon_area
	(
		const solidpolygon&					poly,		// < Solid polygon
		const std::vector<datapoints>&		contours	// < Contours data
	);

//---------------------------------------------------------------------------------------//
//	FUNCION NAME	:	[compute_polygon_intersection_area]
//  MADE BY:			[JSC] Jefferson Salcedo Chavez				DATE: FEB/2026
//	DESCRIPTION		:	Computes the intersection area between two solid polygons
//---------------------------------------------------------------------------------------//
polygon_intersection_result
	compute_polygon_intersection_area
	(
		const solidpolygon&					poly1,			// < First solid polygon
		const std::vector<datapoints>&		contours1,		// < Contours of first polygon
		const solidpolygon&					poly2,			// < Second solid polygon
		const std::vector<datapoints>&		contours2		// < Contours of second polygon
	);

//---------------------------------------------------------------------------------------//
//	FUNCION NAME	:	[format_polygon_name_l1]
//  MADE BY:			[JSC] Jefferson Salcedo Chavez				DATE: FEB/2026
//	DESCRIPTION		:	Formats the name of a L1 polygon using letters
//---------------------------------------------------------------------------------------//
std::string
	format_polygon_name_l1
	(
		const polygon_reference*				ref,			// < Reference to the polygon
		const std::vector<solidpolygon>&		all_polygons,	// < All polygons array
		const std::vector<tree_node>&			forest			// < Forest structure
	);

//---------------------------------------------------------------------------------------//
//	FUNCION NAME	:	[format_polygon_name_l2]
//  MADE BY:			[JSC] Jefferson Salcedo Chavez				DATE: FEB/2026
//	DESCRIPTION		:	Formats the name of a L2 polygon using numbers
//---------------------------------------------------------------------------------------//
std::string
	format_polygon_name_l2
	(
		const polygon_reference*				ref,			// < Reference to the polygon
		const std::vector<solidpolygon>&		all_polygons,	// < All polygons array
		const std::vector<tree_node>&			forest			// < Forest structure
	);

#endif // __POLYGON_H__