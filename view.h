

#ifndef __VIEW_H__
#define __VIEW_H__

#include <string>
#include <vector>
#include <Eigen/Core>
#include <igl/opengl/glfw/Viewer.h>
#include <igl/opengl/glfw/imgui/ImGuiPlugin.h>
#include <igl/opengl/glfw/imgui/ImGuiMenu.h>
#include <igl/opengl/glfw/imgui/ImGuiHelpers.h>
#include "tree.h"
#include "pointset.h"

//---------------------------------------------------------------------------------------//
//---------- 							D E F I N E S 							---------//
//---------------------------------------------------------------------------------------//

//---------------------------------------------------------------------------------------//
//---------					  D A T A   T Y P E   D E C L S				        ---------//
//---------------------------------------------------------------------------------------//

//---------------------------------------------------------------------------------------//
//	STRUCTURE		:	[label3d]
//	MADE BY			:	[JSC] Jefferson Salcedo Chavez		DATE:	FEB/2026
//	DESCRIPTION		:	Almacena la posicion y el texto de una etiqueta 3D
//---------------------------------------------------------------------------------------//
struct label3d {
	Eigen::Vector3f position;
	std::string text;
};

//---------------------------------------------------------------------------------------//
//	FUNCION NAME	:	[projectToScreen]
//  MADE BY:			[JSC] Jefferson Salcedo Chavez				DATE: FEB/2026
//	DESCRIPTION		:	Projects a 3D point to 2D screen coordinates using the viewer
//---------------------------------------------------------------------------------------//
Eigen::Vector2f
	projectToScreen
	(
		const Eigen::Vector3f&				p,			// < 3D point to project
		const igl::opengl::glfw::Viewer&	viewer		// < Viewer for projection matrices
	);

//---------------------------------------------------------------------------------------//
//	FUNCION NAME	:	[layoutTreeForest]
//  MADE BY:			[JSC] Jefferson Salcedo Chavez				DATE: FEB/2026
//	DESCRIPTION		:	Computes the layout positions of tree nodes for diagram drawing
//---------------------------------------------------------------------------------------//
void
	layoutTreeForest
	(
		const std::vector<tree_node>&	forest,			// < Forest structure
		int								node,			// < Current node index
		std::vector<ImVec2>&			pos,			// <> Positions output
		float							levelSpacing,	// < Vertical spacing between levels
		float							siblingSpacing,	// < Horizontal spacing between siblings
		float							currentY,		// < Current Y position
		float&							nextX,			// <> Next available X position
		float							originX,		// < X origin offset
		float							originY			// < Y origin offset
	);

//---------------------------------------------------------------------------------------//
//	FUNCION NAME	:	[drawForestDiagramForest]
//  MADE BY:			[JSC] Jefferson Salcedo Chavez				DATE: FEB/2026
//	DESCRIPTION		:	Draws the forest diagram using ImGui
//---------------------------------------------------------------------------------------//
void
	drawForestDiagramForest
	(
		const std::vector<tree_node>&	forest,			// < Forest structure
		bool							useLetters		// < Use letters (true) or numbers (false)
	);

//---------------------------------------------------------------------------------------//
//	FUNCION NAME	:	[draw_contours_intersection]
//  MADE BY:			[JSC] Jefferson Salcedo Chavez				DATE: FEB/2026
//	DESCRIPTION		:	Draws contours as edges in the viewer with a given color
//---------------------------------------------------------------------------------------//
void
	draw_contours_intersection
	(
		igl::opengl::glfw::Viewer&			viewer,		// <> Viewer to draw on
		const std::vector<datapoints>&		contours,	// < Contours to draw
		const Eigen::RowVector3d&			color		// < Color for the edges
	);

#endif