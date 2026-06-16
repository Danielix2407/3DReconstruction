#define _USE_MATH_DEFINES

#include <iostream>
#include <fstream>
#include <limits>
#include <string>
#include <vector>
#include <utility>
#include <iomanip>
#include <sstream>
#include <filesystem>
#include <igl/opengl/glfw/Viewer.h>
#include <igl/opengl/glfw/imgui/ImGuiPlugin.h>
#include <igl/opengl/glfw/imgui/ImGuiMenu.h>
#include <igl/opengl/glfw/imgui/ImGuiHelpers.h>
#include <igl/unproject_onto_mesh.h>
#include <igl/writeOBJ.h>
#include <igl/readOBJ.h>
#include <igl/boundary_facets.h>
#include <Eigen/Core>
#include <queue>
#include <deque>
#include <unordered_set>
#include <unordered_map>
#include <set>
#include <map>
#include <Eigen/Dense>
#include "pointset.h"
#include "tree.h"
#include "polygon.h"
#include "mapping.h"
#include "level.h"
#include "view.h"
#include "manifold_check.h"
#include <geogram/basic/common.h>
#include <geogram/basic/geometry.h>
#include <geogram/basic/logger.h>
#include <geogram/delaunay/CDT_2d.h>
#include <geogram/delaunay/delaunay.h>

using namespace GEO;

// =======================
// VIEWER FLAGS (true = enabled, false = disabled)
// =======================
constexpr bool ENABLE_VIEWER_L1            = true;
constexpr bool ENABLE_VIEWER_L2            = true;
constexpr bool ENABLE_VIEWER_INTERSECTION  = true;
constexpr bool ENABLE_VIEWER_OVERLAY       = true;
constexpr bool ENABLE_VIEWER_HH            = true;
constexpr bool ENABLE_VIEWER_HS            = true;
constexpr bool ENABLE_VIEWER_VOR_L1        = false;
constexpr bool ENABLE_VIEWER_VOR_L2        = false;
constexpr bool ENABLE_VIEWER_IMA_L1        = false;
constexpr bool ENABLE_VIEWER_EMA_L1        = false;
constexpr bool ENABLE_VIEWER_IMA_L2        = false;
constexpr bool ENABLE_VIEWER_EMA_L2        = false;
constexpr bool ENABLE_VIEWER_BOTH_L1       = false;
constexpr bool ENABLE_VIEWER_BOTH_L2       = false;
constexpr bool ENABLE_VIEWER_PROJ_L1_ON_L2 = false;
constexpr bool ENABLE_VIEWER_PROJ_L2_ON_L1 = false;
constexpr bool ENABLE_VIEWER_DELAUNAY3D    = false;
constexpr bool ENABLE_VIEWER_T1            = false;
constexpr bool ENABLE_VIEWER_T2            = false;
constexpr bool ENABLE_VIEWER_T12           = false;
constexpr bool ENABLE_VIEWER_T1_VALID      = true;
constexpr bool ENABLE_VIEWER_T1_REMOVED    = false;
constexpr bool ENABLE_VIEWER_T2_REMOVED    = false;
constexpr bool ENABLE_VIEWER_ALL_VALID     = true;
constexpr bool ENABLE_VIEWER_ALL_VALID_GRAY = true;
constexpr bool ENABLE_VIEWER_FREE_BOUNDARY  = true;
constexpr bool ENABLE_VIEWER_T12_VALID     = true;
constexpr bool ENABLE_VIEWER_T12_REMOVED   = true;
constexpr bool ENABLE_VIEWER_INTERACTIVE   = true;
constexpr bool ENABLE_VIEWER_T12_INSPECT  = true;
constexpr bool ENABLE_VIEWER_OBJ_RESULT   = true;

// =======================
// IMPLEMENTACIONES
// =======================

bool pointInPolygon(
	const Eigen::Vector3d& p,
	const std::vector<Eigen::Vector3d>& poly)
{
	bool inside = false;

	for (size_t i = 0, j = poly.size() - 1; i < poly.size(); j = i++)
	{
		double xi = poly[i].x();
		double yi = poly[i].y();
		double xj = poly[j].x();
		double yj = poly[j].y();

		bool intersect =
			((yi > p.y()) != (yj > p.y())) &&
			(p.x() < (xj - xi) * (p.y() - yi) / (yj - yi + 1e-15) + xi);

		if (intersect) inside = !inside;
	}

	return inside;
}

int contourDepth(int node, const std::vector<tree_node>& forest)
{
	int depth = 0;
	int current = node;

	while (forest[current].parent != -1)
	{
		depth++;
		current = forest[current].parent;
	}

	return depth;
}

void printTree(
	const std::vector<tree_node>& forest,
	int node,
	int depth)
{
	for (int i = 0; i < depth - 1; ++i)
		std::cout << "  ";

	if (depth > 0)
		std::cout << "|- ";

	std::cout << "Contour " << forest[node].contourId
		<< " (depth=" << forest[node].depth << ")\n";

	for (int c : forest[node].children) {
		printTree(forest, c, depth + 1);
	}
}

double polygonArea(const std::vector<Eigen::Vector3d>& poly)
{
	double area = 0.0;

	for (size_t i = 0, j = poly.size() - 1; i < poly.size(); j = i++)
	{
		area += (poly[j].x() + poly[i].x()) *
			(poly[j].y() - poly[i].y());
	}

	return std::abs(area) * 0.5;
}

void layoutTreeForest(
	const std::vector<tree_node>& forest,
	int node,
	std::vector<ImVec2>& pos,
	float levelSpacing,
	float siblingSpacing,
	float currentY,
	float& nextX,
	float originX,
	float originY)
{
	const auto& n = forest[node];

	float x;
	float y = originY + currentY;

	if (n.children.empty()) {
		x = nextX;
		nextX += siblingSpacing;
	}
	else {
		float left = nextX;

		for (int c : n.children) {
			layoutTreeForest(
				forest, c, pos,
				levelSpacing,
				siblingSpacing,
				currentY + levelSpacing,
				nextX,
				originX,
				originY
			);
		}

		float right = nextX - siblingSpacing;
		x = (left + right) * 0.5f;
	}

	pos[node] = ImVec2(originX + x, y);
}

void drawForestDiagramForest(const std::vector<tree_node>& forest, bool useLetters)
{
	if (forest.empty()) return;

	ImGuiWindowFlags window_flags =
		ImGuiWindowFlags_NoMove |
		ImGuiWindowFlags_NoScrollbar |
		ImGuiWindowFlags_NoCollapse |
		ImGuiWindowFlags_NoBringToFrontOnFocus;

	ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(1.0f, 1.0f, 1.0f, 1.0f));
	ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.0f, 0.0f, 0.0f, 1.0f));
	ImGui::PushStyleColor(ImGuiCol_TitleBg, ImVec4(0.85f, 0.85f, 0.85f, 1.0f));
	ImGui::PushStyleColor(ImGuiCol_TitleBgActive, ImVec4(0.75f, 0.75f, 0.75f, 1.0f));

	ImGui::Begin("Forest Diagram", nullptr, window_flags);

	ImDrawList* draw = ImGui::GetWindowDrawList();
	ImVec2 origin = ImGui::GetCursorScreenPos();
	ImVec2 winSize = ImGui::GetContentRegionAvail();

	draw->AddRectFilled(
		origin,
		ImVec2(origin.x + winSize.x, origin.y + winSize.y),
		IM_COL32(255, 255, 255, 255)
	);

	origin.x += 50;
	origin.y += 50;

	float levelSpacing = 80.0f;
	float siblingSpacing = 60.0f;
	float treeSpacing = 100.0f;

	std::vector<ImVec2> pos(forest.size());

	std::vector<int> roots;
	for (size_t i = 0; i < forest.size(); ++i) {
		if (forest[i].parent == -1)
			roots.push_back(i);
	}

	float nextX = 0.0f;

	for (int root : roots) {
		layoutTreeForest(
			forest,
			root,
			pos,
			levelSpacing,
			siblingSpacing,
			0.0f,
			nextX,
			origin.x,
			origin.y
		);

		nextX += treeSpacing;
	}

	for (size_t i = 0; i < forest.size(); ++i) {
		if (forest[i].parent != -1) {
			draw->AddLine(
				pos[forest[i].parent],
				pos[i],
				IM_COL32(0, 0, 0, 255),
				2.0f
			);
		}
	}

	float nodeRadius = 15.0f;

	for (size_t i = 0; i < forest.size(); ++i) {

		draw->AddCircleFilled(
			pos[i],
			nodeRadius,
			IM_COL32(180, 200, 255, 255)
		);

		std::string txt;

		if (useLetters) {
			txt = std::string(1, char('A' + forest[i].contourId));
		}
		else {
			txt = std::to_string(forest[i].contourId + 1);
		}

		draw->AddText(
			ImVec2(pos[i].x - 5, pos[i].y - 7),
			IM_COL32(0, 0, 0, 255),
			txt.c_str()
		);
	}

	ImGui::Dummy(ImVec2(nextX + 50, roots.size() * levelSpacing + 100));

	ImGui::End();

	ImGui::PopStyleColor(4);
}

bool point_in_polygon_2d_geogram(
	const Eigen::Vector2d& p,
	const std::vector<Eigen::Vector3d>& poly)
{
	bool inside = false;
	size_t n = poly.size();
	if (n < 3) return false;

	for (size_t i = 0, j = n - 1; i < n; j = i++) {
		const auto& pi = poly[i];
		const auto& pj = poly[j];

		bool intersect =
			((pi.y() > p.y()) != (pj.y() > p.y())) &&
			(p.x() < (pj.x() - pi.x()) * (p.y() - pi.y()) /
				(pj.y() - pi.y() + 1e-30) + pi.x());

		if (intersect)
			inside = !inside;
	}
	return inside;
}

void draw_contours_intersection(
	igl::opengl::glfw::Viewer& viewer,
	const std::vector<datapoints>& contours,
	const Eigen::RowVector3d& color)
{
	for (const auto& contour : contours)
		for (size_t i = 0; i < contour.points.size(); ++i) {
			const auto& p0 = contour.points[i];
			const auto& p1 = contour.points[(i + 1) % contour.points.size()];
			viewer.data().add_edges(
				Eigen::RowVector3d(p0.x(), p0.y(), 0),
				Eigen::RowVector3d(p1.x(), p1.y(), 0),
				color
			);
		}
}

double compute_polygon_area(
	const solidpolygon& poly,
	const std::vector<datapoints>& contours)
{
	if (poly.contours.empty()) return 0.0;

	double area = polygonArea(contours[poly.contours[0]].points);

	for (size_t i = 1; i < poly.contours.size(); ++i) {
		area -= polygonArea(contours[poly.contours[i]].points);
	}

	return area;
}

polygon_intersection_result compute_polygon_intersection_area(
	const solidpolygon& poly1,
	const std::vector<datapoints>& contours1,
	const solidpolygon& poly2,
	const std::vector<datapoints>& contours2)
{
	polygon_intersection_result result;
	result.intersection_area = 0.0;
	result.vertices.resize(0, 3);
	result.faces.resize(0, 3);

	CDT2d cdt;

	double minX = std::numeric_limits<double>::infinity();
	double minY = std::numeric_limits<double>::infinity();
	double maxX = -std::numeric_limits<double>::infinity();
	double maxY = -std::numeric_limits<double>::infinity();

	auto expand = [&](const solidpolygon& poly, const std::vector<datapoints>& contours) {
		for (int cid : poly.contours) {
			for (const auto& p : contours[cid].points) {
				minX = std::min(minX, p.x());
				minY = std::min(minY, p.y());
				maxX = std::max(maxX, p.x());
				maxY = std::max(maxY, p.y());
			}
		}
	};

	expand(poly1, contours1);
	expand(poly2, contours2);

	double margin = 0.1 * std::max(maxX - minX, maxY - minY);
	cdt.create_enclosing_rectangle(
		minX - margin, minY - margin,
		maxX + margin, maxY + margin
	);

	auto insert_polygon = [&](const solidpolygon& poly, const std::vector<datapoints>& contours) {
		for (int cid : poly.contours) {
			std::vector<index_t> ids;
			for (const auto& p : contours[cid].points)
				ids.push_back(cdt.insert(vec2(p.x(), p.y())));

			for (size_t i = 0; i < ids.size(); ++i)
				cdt.insert_constraint(ids[i], ids[(i + 1) % ids.size()]);
		}
	};

	insert_polygon(poly1, contours1);
	insert_polygon(poly2, contours2);
	cdt.remove_external_triangles();

	for (index_t t = 0; t < cdt.nT(); ++t) {
		vec2 a = cdt.point(cdt.Tv(t, 0));
		vec2 b = cdt.point(cdt.Tv(t, 1));
		vec2 c = cdt.point(cdt.Tv(t, 2));

		Eigen::Vector2d bary((a.x + b.x + c.x) / 3.0, (a.y + b.y + c.y) / 3.0);

		int count1 = 0;
		for (int cid : poly1.contours) {
			if (point_in_polygon_2d_geogram(bary, contours1[cid].points))
				count1++;
		}
		bool in_poly1 = (count1 % 2) == 1;

		int count2 = 0;
		for (int cid : poly2.contours) {
			if (point_in_polygon_2d_geogram(bary, contours2[cid].points))
				count2++;
		}
		bool in_poly2 = (count2 % 2) == 1;

		if (in_poly1 && in_poly2) {
			result.intersection_area += Geom::triangle_area(a, b, c);

			int base = result.vertices.rows();
			result.vertices.conservativeResize(base + 3, 3);
			result.faces.conservativeResize(result.faces.rows() + 1, 3);

			result.vertices.row(base + 0) << a.x, a.y, 0;
			result.vertices.row(base + 1) << b.x, b.y, 0;
			result.vertices.row(base + 2) << c.x, c.y, 0;

			result.faces.row(result.faces.rows() - 1) << base, base + 1, base + 2;
		}
	}

	return result;
}

std::string format_polygon_name_l1(
	const polygon_reference* ref,
	const std::vector<solidpolygon>& all_polygons,
	const std::vector<tree_node>& forest)
{
	const auto& poly = all_polygons[ref->polygon_id];
	std::string result = "(";

	for (size_t i = 0; i < poly.contours.size(); ++i) {
		int cid = poly.contours[i];
		char letter = 'A' + forest[cid].contourId;
		result += letter;
		if (i + 1 < poly.contours.size())
			result += ",";
	}
	result += ")";
	return result;
}

std::string format_polygon_name_l2(
	const polygon_reference* ref,
	const std::vector<solidpolygon>& all_polygons,
	const std::vector<tree_node>& forest)
{
	const auto& poly = all_polygons[ref->polygon_id];
	std::string result = "(";

	for (size_t i = 0; i < poly.contours.size(); ++i) {
		int cid = poly.contours[i];
		int number = forest[cid].contourId + 1;
		result += std::to_string(number);
		if (i + 1 < poly.contours.size())
			result += ",";
	}
	result += ")";
	return result;
}

//---------------------------------------------------------------------------------------//
//	FUNCTION		:	[format_tree_structure]
//	DESCRIPTION		:	Ã¢Å“â€¦ NUEVA: Formatea recursivamente la estructura de un ÃƒÂ¡rbol
//---------------------------------------------------------------------------------------//
std::string format_tree_structure(
	const std::vector<tree_node>& forest,
	int node,
	bool useLetters)
{
	std::string result;

	if (useLetters) {
		result += char('A' + forest[node].contourId);
	}
	else {
		result += std::to_string(forest[node].contourId + 1);
	}

	for (size_t i = 0; i < forest[node].children.size(); ++i) {
		int child = forest[node].children[i];
		result += " ";
		if (!forest[child].children.empty()) {
			result += "(";
			result += format_tree_structure(forest, child, useLetters);
			result += ")";
		}
		else {
			result += format_tree_structure(forest, child, useLetters);
		}
	}

	return result;
}

//---------------------------------------------------------------------------------------//
//	FUNCTION		:	[format_forest_text]
//	DESCRIPTION		:	Ã¢Å“â€¦ NUEVA: Formatea todo el bosque en texto
//---------------------------------------------------------------------------------------//
std::string format_forest_text(
	const std::vector<tree_node>& forest,
	bool useLetters)
{
	if (forest.empty()) return "";

	std::vector<int> roots;
	for (size_t i = 0; i < forest.size(); ++i) {
		if (forest[i].parent == -1)
			roots.push_back(i);
	}

	std::string result = "{";
	for (size_t i = 0; i < roots.size(); ++i) {
		result += "(" + format_tree_structure(forest, roots[i], useLetters) + ")";
		if (i + 1 < roots.size())
			result += ",";
	}
	result += "}";

	return result;
}

std::vector<similarity_matrix_entry> compute_similarity_matrix(
	const std::vector<polygon_reference>& refs_l1,
	const std::vector<polygon_reference>& refs_l2,
	const std::vector<solidpolygon>& all_polygons,
	const std::vector<datapoints>& contours_l1,
	const std::vector<datapoints>& contours_l2)
{
	std::vector<similarity_matrix_entry> matrix;

	for (size_t i = 0; i < refs_l1.size(); ++i) {
		for (size_t j = 0; j < refs_l2.size(); ++j) {

			const auto& ref_l1 = refs_l1[i];
			const auto& ref_l2 = refs_l2[j];

			const auto& poly_l1 = all_polygons[ref_l1.polygon_id];
			const auto& poly_l2 = all_polygons[ref_l2.polygon_id];

			double area_l1 = compute_polygon_area(poly_l1, contours_l1);
			double area_l2 = compute_polygon_area(poly_l2, contours_l2);

			polygon_intersection_result inter = compute_polygon_intersection_area(
				poly_l1, contours_l1,
				poly_l2, contours_l2
			);

			double similarity = 0.0;
			if (area_l1 > 1e-10 && area_l2 > 1e-10 && inter.intersection_area > 1e-10) {
				double ratio1 = inter.intersection_area / area_l1;
				double ratio2 = inter.intersection_area / area_l2;
				similarity = std::min(ratio1, ratio2);
			}

			similarity_matrix_entry entry;
			entry.poly_l1_ref = &refs_l1[i];
			entry.poly_l2_ref = &refs_l2[j];
			entry.similarity_index = similarity;
			entry.intersection_area = inter.intersection_area;
			entry.area_l1 = area_l1;
			entry.area_l2 = area_l2;
			entry.vertices = inter.vertices;
			entry.faces = inter.faces;

			matrix.push_back(entry);
		}
	}

	return matrix;
}

std::vector<mapping_group> compute_mapping_groups(
	const std::vector<solidpolygon>& polygons_l1,
	const std::vector<datapoints>& contours_l1,
	const std::vector<solidpolygon>& polygons_l2,
	const std::vector<datapoints>& contours_l2,
	double threshold)
{
	std::vector<mapping_group> result;

	std::unordered_set<int> S1;
	std::unordered_set<int> S2;

	for (size_t i = 0; i < polygons_l1.size(); ++i)
		S1.insert(i);
	for (size_t i = 0; i < polygons_l2.size(); ++i)
		S2.insert(i);

	while (!S1.empty() || !S2.empty()) {

		mapping_group group;
		std::queue<polygon_pair> queue;

		if (!S1.empty()) {
			int seed_id = *S1.begin();
			S1.erase(seed_id);

			group.polygons_l1.push_back(seed_id);
			queue.push(polygon_pair(polygon_reference(seed_id, 0)));

		}
		else {
			int seed_id = *S2.begin();
			S2.erase(seed_id);

			group.polygons_l2.push_back(seed_id);
			queue.push(polygon_pair(polygon_reference(seed_id, 1)));
		}

		while (!queue.empty()) {

			polygon_pair current = queue.front();
			queue.pop();

			if (current.ref.level == 0) {
				double area_group_l1 = 0.0;
				for (int pid : group.polygons_l1) {
					area_group_l1 += compute_polygon_area(polygons_l1[pid], contours_l1);
				}

				std::vector<int> s2_copy(S2.begin(), S2.end());

				for (int q_id : s2_copy) {
					double intersection_area = 0.0;

					for (int pid : group.polygons_l1) {
						auto inter = compute_polygon_intersection_area(
							polygons_l1[pid], contours_l1,
							polygons_l2[q_id], contours_l2
						);
						intersection_area += inter.intersection_area;
					}

					double area_q = compute_polygon_area(polygons_l2[q_id], contours_l2);

					if (area_group_l1 < 1e-10 || area_q < 1e-10) continue;

					double ratio1 = intersection_area / area_group_l1;
					double ratio2 = intersection_area / area_q;
					double similarity = std::min(ratio1, ratio2);

					if (similarity > threshold) {
						queue.push(polygon_pair(polygon_reference(q_id, 1)));
						group.polygons_l2.push_back(q_id);
						S2.erase(q_id);
					}
				}

			}
			else {
				double area_group_l2 = 0.0;
				for (int pid : group.polygons_l2) {
					area_group_l2 += compute_polygon_area(polygons_l2[pid], contours_l2);
				}

				std::vector<int> s1_copy(S1.begin(), S1.end());

				for (int q_id : s1_copy) {
					double intersection_area = 0.0;

					for (int pid : group.polygons_l2) {
						auto inter = compute_polygon_intersection_area(
							polygons_l2[pid], contours_l2,
							polygons_l1[q_id], contours_l1
						);
						intersection_area += inter.intersection_area;
					}

					double area_q = compute_polygon_area(polygons_l1[q_id], contours_l1);

					if (area_group_l2 < 1e-10 || area_q < 1e-10) continue;

					double ratio1 = intersection_area / area_group_l2;
					double ratio2 = intersection_area / area_q;
					double similarity = std::min(ratio1, ratio2);

					if (similarity > threshold) {
						queue.push(polygon_pair(polygon_reference(q_id, 0)));
						group.polygons_l1.push_back(q_id);
						S1.erase(q_id);
					}
				}
			}
		}

		result.push_back(group);
	}

	return result;
}

std::vector<datapoints> resample_contours(
	const std::vector<datapoints>& contours,
	int factor)
{
	if (factor <= 1) return contours;

	std::vector<datapoints> result;
	for (const auto& contour : contours) {
		datapoints resampled;
		const auto& pts = contour.points;
		for (size_t i = 0; i < pts.size(); ++i) {
			const auto& p0 = pts[i];
			const auto& p1 = pts[(i + 1) % pts.size()];
			resampled.points.push_back(p0);
			for (int k = 1; k < factor; ++k) {
				double t = static_cast<double>(k) / factor;
				Eigen::Vector3d mid = p0 + t * (p1 - p0);
				resampled.points.push_back(mid);
			}
		}
		result.push_back(resampled);
	}
	for (auto& c : result)
		c.compute_bbox();
	return result;
}

std::vector<std::pair<vec2, vec2>> compute_voronoi_edges(
	const std::vector<datapoints>& contours)
{
	std::vector<double> coords;
	for (const auto& c : contours)
		for (const auto& p : c.points) {
			coords.push_back(p.x());
			coords.push_back(p.y());
		}

	std::vector<std::pair<vec2, vec2>> vor_edges;
	index_t N = coords.size() / 2;
	if (N < 3) return vor_edges;

	Delaunay_var delaunay = Delaunay::create(2, "BDEL2d");
	delaunay->set_vertices(N, coords.data());

	index_t nb_cells = delaunay->nb_cells();
	std::vector<vec2> centers(nb_cells);
	for (index_t t = 0; t < nb_cells; ++t) {
		const signed_index_t* v = delaunay->cell_to_v() + 3 * t;
		vec2 p0(delaunay->vertex_ptr(v[0]));
		vec2 p1(delaunay->vertex_ptr(v[1]));
		vec2 p2(delaunay->vertex_ptr(v[2]));
		centers[t] = Geom::triangle_circumcenter(p0, p1, p2);
	}

	for (index_t t = 0; t < nb_cells; ++t)
		for (index_t lf = 0; lf < 3; ++lf) {
			signed_index_t adj = delaunay->cell_adjacent(t, lf);
			if (adj != NO_INDEX && t < index_t(adj))
				vor_edges.emplace_back(centers[t], centers[adj]);
		}

	return vor_edges;
}

int classify_point_solid_polygon(
	const Eigen::Vector2d& p,
	const std::vector<polygon_reference>& poly_refs,
	const std::vector<solidpolygon>& all_polys,
	const std::vector<datapoints>& contours)
{
	for (size_t idx = 0; idx < poly_refs.size(); ++idx) {
		const auto& poly = all_polys[poly_refs[idx].polygon_id];
		if (poly.contours.empty()) continue;

		int count = 0;
		for (int cid : poly.contours) {
			const auto& c = contours[cid];
			if (c.bbox_valid &&
				(p.x() < c.bbox_min_x || p.x() > c.bbox_max_x ||
				 p.y() < c.bbox_min_y || p.y() > c.bbox_max_y))
				continue;
			if (point_in_polygon_2d_geogram(p, c.points))
				count++;
		}
		if (count % 2 == 1)
			return static_cast<int>(idx);
	}
	return -1;
}

bool point_near_contour_boundary(
	const Eigen::Vector2d& p,
	const std::vector<polygon_reference>& poly_refs,
	const std::vector<solidpolygon>& all_polys,
	const std::vector<datapoints>& contours,
	double tol = 1e-6)
{
	double tol2 = tol * tol;
	for (const auto& ref : poly_refs) {
		const auto& poly = all_polys[ref.polygon_id];
		for (int cid : poly.contours) {
			const auto& c = contours[cid];
			if (c.bbox_valid &&
				(p.x() < c.bbox_min_x - tol || p.x() > c.bbox_max_x + tol ||
				 p.y() < c.bbox_min_y - tol || p.y() > c.bbox_max_y + tol))
				continue;
			const auto& pts = c.points;
			size_t n = pts.size();
			for (size_t i = 0; i < n; ++i) {
				size_t j = (i + 1) % n;
				double ax = pts[i].x(), ay = pts[i].y();
				double bx = pts[j].x(), by = pts[j].y();
				double dx = bx - ax, dy = by - ay;
				double len2 = dx * dx + dy * dy;
				if (len2 < 1e-30) continue;
				double t = ((p.x() - ax) * dx + (p.y() - ay) * dy) / len2;
				if (t < 0.0) t = 0.0;
				if (t > 1.0) t = 1.0;
				double cx = ax + t * dx - p.x();
				double cy = ay + t * dy - p.y();
				if (cx * cx + cy * cy <= tol2)
					return true;
			}
		}
	}
	return false;
}

struct medial_axes_result {
	std::vector<std::pair<vec2, vec2>> ima_edges;
	std::vector<std::pair<vec2, vec2>> ema_edges;
	std::vector<std::pair<vec2, vec2>> discarded_edges;
};

// A polygon after removing already-used contours during 2D Shape Similarity depuration
struct partial_polygon_entry {
	std::vector<int> contour_ids; // remaining contour indices (into contours_l1 or contours_l2)
};

// A final (depurated) mapping group produced by the 2D Shape Similarity algorithm
struct final_mapping_group_2d {
	std::vector<partial_polygon_entry> l1_side;
	std::vector<partial_polygon_entry> l2_side;
	int level;
	bool is_solid; // true = solid polygon group, false = hole polygon group
};

struct SolidRegionGrid {
    // Sparse storage: map from cell (ix,iy) to region id. Avoid allocating dense nx*ny
    // grid which can be huge when contours are far apart.
    std::unordered_map<long long, int> data_map;
	double min_x, min_y, cell_w, cell_h;
	int nx, ny;
	bool valid = false;

	// Stored pointers for direct classification when precompute is disabled
	const std::vector<polygon_reference>* stored_poly_refs = nullptr;
	const std::vector<solidpolygon>* stored_all_polys = nullptr;
	const std::vector<datapoints>* stored_contours = nullptr;

	void precompute(
		const std::vector<polygon_reference>& poly_refs,
		const std::vector<solidpolygon>& all_polys,
		const std::vector<datapoints>& contours,
		int resolution = 1000)
	{
		// Simplified: do not build a global grid or bounding box here. Instead
		// store pointers to the polygon data so classify() can perform exact
		// point-in-polygon tests on demand. This avoids constructing huge grids
		// when contours are far apart while preserving correctness.
		stored_poly_refs = &poly_refs;
		stored_all_polys = &all_polys;
		stored_contours = &contours;
		data_map.clear();
		nx = ny = 0; cell_w = cell_h = 0.0;
		valid = true;
	}

    int classify(double x, double y) const {
        if (!valid) return -1;
        // If no raster grid was built, fall back to exact point-in-polygon test.
        if (nx <= 0 || cell_w == 0.0 || cell_h == 0.0) {
            if (stored_poly_refs && stored_all_polys && stored_contours) {
                Eigen::Vector2d p(x, y);
                return classify_point_solid_polygon(p, *stored_poly_refs, *stored_all_polys, *stored_contours);
            }
            return -1;
        }

        int ix = static_cast<int>((x - min_x) / cell_w);
        int iy = static_cast<int>((y - min_y) / cell_h);
        if (ix < 0 || ix >= nx || iy < 0 || iy >= ny) return -1;
        long long key = (static_cast<long long>(iy) << 32) | static_cast<unsigned int>(ix);
        auto it = data_map.find(key);
        if (it != data_map.end()) return it->second;
        // Not present in sparse map: fallback to exact test
        if (stored_poly_refs && stored_all_polys && stored_contours) {
            Eigen::Vector2d p(x, y);
            return classify_point_solid_polygon(p, *stored_poly_refs, *stored_all_polys, *stored_contours);
        }
        return -1;
    }
};

medial_axes_result classify_voronoi_edges(
	const std::vector<std::pair<vec2, vec2>>& vor_edges,
	const std::vector<polygon_reference>& poly_refs,
	const std::vector<solidpolygon>& all_polys,
	const std::vector<datapoints>& contours,
	const SolidRegionGrid* grid = nullptr)
{
	medial_axes_result result;

	for (const auto& edge : vor_edges) {
		Eigen::Vector2d p1(edge.first.x, edge.first.y);
		Eigen::Vector2d p2(edge.second.x, edge.second.y);

		int region1 = grid ? grid->classify(p1.x(), p1.y()) : classify_point_solid_polygon(p1, poly_refs, all_polys, contours);
		int region2 = grid ? grid->classify(p2.x(), p2.y()) : classify_point_solid_polygon(p2, poly_refs, all_polys, contours);

		bool in_solid_1 = (region1 >= 0);
		bool in_solid_2 = (region2 >= 0);

		if (in_solid_1 && in_solid_2) {
			if (region1 == region2) {
				result.ima_edges.push_back(edge);
			} else {
				result.discarded_edges.push_back(edge);
			}
		}
		else if (!in_solid_1 && !in_solid_2) {
			result.ema_edges.push_back(edge);
		}
		else {
			result.discarded_edges.push_back(edge);
		}
	}

	return result;
}

bool segment_intersect_2d(
	double ax, double ay, double bx, double by,
	double cx, double cy, double dx, double dy,
	double& t)
{
	double denom = (bx - ax) * (dy - cy) - (by - ay) * (dx - cx);
	if (std::abs(denom) < 1e-15) return false;

	double t_num = (cx - ax) * (dy - cy) - (cy - ay) * (dx - cx);
	double u_num = (cx - ax) * (by - ay) - (cy - ay) * (bx - ax);

	t = t_num / denom;
	double u = u_num / denom;

	return (t > 1e-10 && t < 1.0 - 1e-10 && u > -1e-10 && u < 1.0 + 1e-10);
}

bool find_closest_contour_intersection(
	const Eigen::Vector2d& p_solid,
	const Eigen::Vector2d& p_void,
	const std::vector<polygon_reference>& poly_refs,
	const std::vector<solidpolygon>& all_polys,
	const std::vector<datapoints>& contours,
	Eigen::Vector2d& hit_point)
{
	double best_t = 2.0;
	bool found = false;

	for (const auto& ref : poly_refs) {
		const auto& poly = all_polys[ref.polygon_id];
		for (int cid : poly.contours) {
			const auto& pts = contours[cid].points;
			for (size_t i = 0; i < pts.size(); ++i) {
				size_t j = (i + 1) % pts.size();
				double t;
				if (segment_intersect_2d(
					p_solid.x(), p_solid.y(), p_void.x(), p_void.y(),
					pts[i].x(), pts[i].y(), pts[j].x(), pts[j].y(), t))
				{
					if (t < best_t) {
						best_t = t;
						found = true;
					}
				}
			}
		}
	}

	if (found) {
		hit_point = p_solid + best_t * (p_void - p_solid);
	}
	return found;
}

struct ema_projection_result {
	std::vector<Eigen::Vector2d> solid_vertices;
	std::vector<Eigen::Vector2d> void_vertices;
	std::vector<std::pair<vec2, vec2>> edges_with_solid;
	std::vector<std::pair<vec2, vec2>> edges_without_solid;
};

ema_projection_result compute_ema_projection(
	const std::vector<std::pair<vec2, vec2>>& ema_edges,
	const std::vector<polygon_reference>& target_poly_refs,
	const std::vector<solidpolygon>& target_all_polys,
	const std::vector<datapoints>& target_contours,
	const SolidRegionGrid* grid = nullptr)
{
	ema_projection_result result;
	std::set<std::pair<double,double>> added_solid;
	std::set<std::pair<double,double>> added_void;

	for (const auto& edge : ema_edges) {
		Eigen::Vector2d p1(edge.first.x, edge.first.y);
		Eigen::Vector2d p2(edge.second.x, edge.second.y);

		int r1 = grid ? grid->classify(p1.x(), p1.y()) : classify_point_solid_polygon(p1, target_poly_refs, target_all_polys, target_contours);
		int r2 = grid ? grid->classify(p2.x(), p2.y()) : classify_point_solid_polygon(p2, target_poly_refs, target_all_polys, target_contours);

		bool in1 = (r1 >= 0);
		bool in2 = (r2 >= 0);

		auto key1 = std::make_pair(p1.x(), p1.y());
		auto key2 = std::make_pair(p2.x(), p2.y());

		if (in1) {
			if (added_solid.find(key1) == added_solid.end()) {
				result.solid_vertices.push_back(p1);
				added_solid.insert(key1);
			}
		} else {
			if (added_void.find(key1) == added_void.end()) {
				result.void_vertices.push_back(p1);
				added_void.insert(key1);
			}
		}

		if (in2) {
			if (added_solid.find(key2) == added_solid.end()) {
				result.solid_vertices.push_back(p2);
				added_solid.insert(key2);
			}
		} else {
			if (added_void.find(key2) == added_void.end()) {
				result.void_vertices.push_back(p2);
				added_void.insert(key2);
			}
		}

		if (in1 && in2) {
			result.edges_with_solid.push_back(edge);
		}
		else if (!in1 && !in2) {
			result.edges_without_solid.push_back(edge);
		}
		else {
			Eigen::Vector2d p_solid = in1 ? p1 : p2;
			Eigen::Vector2d p_void  = in1 ? p2 : p1;
			Eigen::Vector2d hit;

			if (find_closest_contour_intersection(
				p_solid, p_void,
				target_poly_refs, target_all_polys, target_contours, hit))
			{
				result.edges_with_solid.emplace_back(
					vec2(p_solid.x(), p_solid.y()),
					vec2(hit.x(), hit.y()));
				result.edges_without_solid.emplace_back(
					vec2(hit.x(), hit.y()),
					vec2(p_void.x(), p_void.y()));
			}
			else {
				result.edges_with_solid.push_back(edge);
			}
		}
	}

	return result;
}

// =======================
// VARIABLES GLOBALES
// =======================
std::vector<tree_node> forest_l1;
std::vector<tree_node> forest_l2;
std::vector<label3d> labels;
std::vector<label3d> labels_l2;

std::vector<solidpolygon> all_polygons;
std::vector<polygon_reference> polygon_refs_l1;
std::vector<polygon_reference> polygon_refs_l2;

std::vector<mapping_group> global_mapping_groups;
std::vector<mapping_group> global_hole_mapping_groups;
std::vector<similarity_matrix_entry> global_similarity_matrix;

std::vector<solidpolygon> all_hole_polygons;
std::vector<polygon_reference> hole_refs_l1;
std::vector<polygon_reference> hole_refs_l2;
std::vector<similarity_matrix_entry> hole_vs_hole_matrix;
std::vector<similarity_matrix_entry> hole_vs_solid_matrix;

std::string global_forest_text_l1;  // Texto del bosque L1
std::string global_forest_text_l2;  // Texto del bosque L2

std::vector<Eigen::Vector3d> augmented_points_l1;
std::vector<Eigen::Vector3d> augmented_points_l2;
int aug_contour_l1 = 0, aug_ima_l1 = 0, aug_ema_proj_l1 = 0;
int aug_contour_l2 = 0, aug_ima_l2 = 0, aug_ema_proj_l2 = 0;
int delaunay3d_T12_valid = 0, delaunay3d_T12_removed = 0;
int delaunay3d_T12_valid_faces = 0, delaunay3d_T12_removed_faces = 0;
int delaunay3d_T1_valid = 0, delaunay3d_T1_removed = 0;
int delaunay3d_T2_valid = 0, delaunay3d_T2_removed = 0;
int delaunay3d_T1_valid_faces = 0, delaunay3d_T1_removed_faces = 0;
int delaunay3d_T2_valid_faces = 0, delaunay3d_T2_removed_faces = 0;
int delaunay3d_all_valid_faces = 0;
int delaunay3d_free_boundary_faces = 0;
manifold_result global_manifold_info;

// Free boundary (skin) faces
std::vector<std::array<index_t, 3>> global_faces_free_boundary;
// Interactive viewer data (global for callback access)
Eigen::MatrixXd global_V3d;
std::vector<std::array<index_t, 3>> global_faces_t1_valid;
std::vector<std::array<index_t, 3>> global_faces_t2_valid;
std::vector<std::array<index_t, 3>> global_faces_t12_valid;
std::vector<std::array<index_t, 3>> global_faces_t1_removed;
std::vector<std::array<index_t, 3>> global_faces_t2_removed;
std::vector<std::array<index_t, 3>> global_faces_t12_removed;
bool interactive_needs_rebuild = true;

// Per-T1/T2-tet filter data for runtime toggling
	struct t1_t2_tet_info {
	int type; // 1 = T1, 2 = T2
	std::array<std::array<index_t, 3>, 4> faces;
	index_t orig_verts[4]; // original vertex order from Geogram cell_to_v()
	index_t base_verts[3];
	index_t apex_vert;
		bool centroid_in_solid;
		bool fails_midpoint;
		bool fails_group_affinity = false; // set by algo4 group affinity filter
	int void_l1_count; // how many sample points are void in L1 (out of 9)
	int void_l2_count; // how many sample points are void in L2 (out of 9)
	bool midpoint_in_solid_l1[3][3]; // per-edge, per-sample: in solid L1?
	bool midpoint_in_solid_l2[3][3]; // per-edge, per-sample: in solid L2?
	double centroid_x, centroid_y;
	double midpoint_x[3][3], midpoint_y[3][3]; // [edge][sample]
};
std::vector<t1_t2_tet_info> global_t1_tets;
std::vector<t1_t2_tet_info> global_t2_tets;
bool global_filter_t1_midpoint = true;
bool global_filter_t2_midpoint = true;
int global_t1_removed_by_centroid = 0;
int global_t1_removed_by_midpoint = 0;
int global_t2_removed_by_centroid = 0;
int global_t2_removed_by_midpoint = 0;

// T1/T2 inspect viewer state
int global_t1t2_inspect_idx = -1;
int global_t1t2_inspect_type = 0; // 1=T1, 2=T2
bool global_t1t2_inspect_is_removed = false;

// Per-T12-tet filter data for runtime toggling
	struct t12_tet_info {
		std::array<std::array<index_t, 3>, 4> faces;
		index_t verts[4];
		bool fails_midpoint;
		bool fails_isolated;
		bool fails_group_affinity = false; // set by algo4 group affinity filter
	};
std::vector<t12_tet_info> global_t12_tets;
bool global_filter_t12_midpoint = true;
bool global_filter_t12_isolated = true;
int global_t12_removed_by_midpoint = 0;
int global_t12_removed_by_isolated = 0;
int global_t12_rescued_by_ima = 0;
bool global_filter_t12_ima_rescue = true;

// IMA vertex index ranges (global indices in the 3D Delaunay point set)
index_t global_ima_start_l1 = 0, global_ima_end_l1 = 0;
index_t global_ima_start_l2 = 0, global_ima_end_l2 = 0;

// Global data needed by IMA rescue segment validation
index_t global_n_l1_pts = 0;
const std::vector<datapoints>* global_filter_contours_l1 = nullptr;
const std::vector<datapoints>* global_filter_contours_l2 = nullptr;
const SolidRegionGrid* global_grid_l1 = nullptr;
const SolidRegionGrid* global_grid_l2 = nullptr;

// T12 inspect viewer state
int global_t12_inspect_idx = -1;
bool global_t12_inspect_is_removed = false;
bool global_t12_inspect_needs_rebuild = true;

// Interactive viewer face data (for mouse picking)
std::vector<std::array<index_t, 3>> global_interactive_faces;
std::vector<int> global_interactive_face_types;

// Multi-section reconstruction data
Eigen::MatrixXd global_V3d_multi;
std::vector<std::array<index_t, 3>> global_accum_faces_t1_valid;
std::vector<std::array<index_t, 3>> global_accum_faces_t2_valid;
std::vector<std::array<index_t, 3>> global_accum_faces_t12_valid;
std::vector<std::array<index_t, 3>> global_all_valid_faces;
int global_multi_range_start = 0, global_multi_range_end = 0;
std::vector<final_mapping_group_2d> global_final_groups_2d;

void recompute_t1_t2_from_filters() {
	global_faces_t1_valid.clear();
	global_faces_t1_removed.clear();
	global_faces_t2_valid.clear();
	global_faces_t2_removed.clear();
	global_t1_removed_by_centroid = 0;
	global_t1_removed_by_midpoint = 0;
	global_t2_removed_by_centroid = 0;
	global_t2_removed_by_midpoint = 0;

	int t1_valid_count = 0, t1_removed_count = 0;
	for (const auto& info : global_t1_tets) {
		bool valid = info.centroid_in_solid;
		if (!valid) global_t1_removed_by_centroid++;
		if (valid && global_filter_t1_midpoint && info.fails_midpoint) {
			valid = false;
			global_t1_removed_by_midpoint++;
		}
		auto& target = valid ? global_faces_t1_valid : global_faces_t1_removed;
		if (valid) t1_valid_count++; else t1_removed_count++;
		for (int lf = 0; lf < 4; ++lf)
			target.push_back(info.faces[lf]);
	}

	int t2_valid_count = 0, t2_removed_count = 0;
	for (const auto& info : global_t2_tets) {
		bool valid = info.centroid_in_solid;
		if (!valid) global_t2_removed_by_centroid++;
		if (valid && global_filter_t2_midpoint && info.fails_midpoint) {
			valid = false;
			global_t2_removed_by_midpoint++;
		}
		auto& target = valid ? global_faces_t2_valid : global_faces_t2_removed;
		if (valid) t2_valid_count++; else t2_removed_count++;
		for (int lf = 0; lf < 4; ++lf)
			target.push_back(info.faces[lf]);
	}

	delaunay3d_T1_valid = t1_valid_count;
	delaunay3d_T1_removed = t1_removed_count;
	delaunay3d_T1_valid_faces = static_cast<int>(global_faces_t1_valid.size());
	delaunay3d_T1_removed_faces = static_cast<int>(global_faces_t1_removed.size());
	delaunay3d_T2_valid = t2_valid_count;
	delaunay3d_T2_removed = t2_removed_count;
	delaunay3d_T2_valid_faces = static_cast<int>(global_faces_t2_valid.size());
	delaunay3d_T2_removed_faces = static_cast<int>(global_faces_t2_removed.size());
}

void recompute_t12_from_filters() {
	// Helper: create a canonical face key (sorted triple)
	auto make_face_key = [](index_t a, index_t b, index_t c) -> std::array<index_t, 3> {
		std::array<index_t, 3> f = {a, b, c};
		if (f[0] > f[1]) std::swap(f[0], f[1]);
		if (f[1] > f[2]) std::swap(f[1], f[2]);
		if (f[0] > f[1]) std::swap(f[0], f[1]);
		return f;
	};

	size_t n_t12 = global_t12_tets.size();

	// Step 1: Determine which T12s survive midpoint (pre-isolated candidates)
	std::vector<bool> survives_pre(n_t12, true);
	global_t12_removed_by_midpoint = 0;

	for (size_t i = 0; i < n_t12; ++i) {
		if (global_filter_t12_midpoint && global_t12_tets[i].fails_midpoint) {
			survives_pre[i] = false;
			global_t12_removed_by_midpoint++;
		}
	}

	// Step 2: Iterative isolated filter on surviving T12s
	std::vector<bool> is_valid(n_t12);
	for (size_t i = 0; i < n_t12; ++i)
		is_valid[i] = survives_pre[i];

	if (global_filter_t12_isolated && n_t12 > 0) {
		// Transitive flood-fill: seed from valid T1/T2 faces and propagate to
		// adjacent T12s, matching the per-group isolated filter behaviour of algo 3.
		std::set<std::array<index_t, 3>> reachable_faces;
		for (const auto& f : global_faces_t1_valid)
			reachable_faces.insert(make_face_key(f[0], f[1], f[2]));
		for (const auto& f : global_faces_t2_valid)
			reachable_faces.insert(make_face_key(f[0], f[1], f[2]));

		bool changed = true;
		while (changed) {
			changed = false;
			for (size_t i = 0; i < n_t12; ++i) {
				if (!is_valid[i]) continue;
				bool touches = false;
				for (int lf = 0; lf < 4; ++lf) {
					auto fk = make_face_key(
						global_t12_tets[i].faces[lf][0],
						global_t12_tets[i].faces[lf][1],
						global_t12_tets[i].faces[lf][2]);
					if (reachable_faces.count(fk)) { touches = true; break; }
				}
				if (touches) {
					// Expose all 4 faces so adjacent T12s can propagate transitively
					for (int lf = 0; lf < 4; ++lf)
						reachable_faces.insert(make_face_key(
							global_t12_tets[i].faces[lf][0],
							global_t12_tets[i].faces[lf][1],
							global_t12_tets[i].faces[lf][2]));
				} else {
					is_valid[i] = false;
					changed = true;
				}
			}
		}
	}

	// Step 2b: IMA rescue â€” T12s removed by isolation that have a vertex on the IMA are rescued
	// Additional check: the segment (L1-L1 or L2-L2) containing the IMA vertex must have
	// its midpoint inside solid region on BOTH L1 and L2.
	global_t12_rescued_by_ima = 0;
	if (global_filter_t12_isolated && global_filter_t12_ima_rescue && n_t12 > 0
		&& global_filter_contours_l1 && global_filter_contours_l2 && global_V3d.rows() > 0) {
		for (size_t i = 0; i < n_t12; ++i) {
			if (survives_pre[i] && !is_valid[i]) {
				// Find which segment has an IMA vertex
				// Separate verts into L1 pair and L2 pair
				index_t vl1[2], vl2[2];
				int cl1 = 0, cl2 = 0;
				for (int lv = 0; lv < 4; ++lv) {
					index_t vi = global_t12_tets[i].verts[lv];
					if (vi < global_n_l1_pts) { if (cl1 < 2) vl1[cl1++] = vi; }
					else { if (cl2 < 2) vl2[cl2++] = vi; }
				}
				if (cl1 != 2 || cl2 != 2) continue;

				// Check if L1 segment has an IMA vertex
				bool l1_has_ima = false;
				for (int k = 0; k < 2; ++k)
					if (vl1[k] >= global_ima_start_l1 && vl1[k] < global_ima_end_l1)
						l1_has_ima = true;

				// Check if L2 segment has an IMA vertex
				bool l2_has_ima = false;
				for (int k = 0; k < 2; ++k)
					if (vl2[k] >= global_ima_start_l2 && vl2[k] < global_ima_end_l2)
						l2_has_ima = true;

				if (!l1_has_ima && !l2_has_ima) continue;

				// For the segment that has the IMA vertex, compute its midpoint
				// and check it falls in solid on BOTH levels
				bool rescue = false;
				if (l1_has_ima) {
					double mx = (global_V3d(vl1[0], 0) + global_V3d(vl1[1], 0)) * 0.5;
					double my = (global_V3d(vl1[0], 1) + global_V3d(vl1[1], 1)) * 0.5;
					bool in_l1 = global_grid_l1 ? (global_grid_l1->classify(mx, my) >= 0) : (classify_point_solid_polygon(Eigen::Vector2d(mx,my), polygon_refs_l1, all_polygons, *global_filter_contours_l1) >= 0);
					bool in_l2 = global_grid_l2 ? (global_grid_l2->classify(mx, my) >= 0) : (classify_point_solid_polygon(Eigen::Vector2d(mx,my), polygon_refs_l2, all_polygons, *global_filter_contours_l2) >= 0);
					if (in_l1 && in_l2) rescue = true;
				}
				if (!rescue && l2_has_ima) {
					double mx = (global_V3d(vl2[0], 0) + global_V3d(vl2[1], 0)) * 0.5;
					double my = (global_V3d(vl2[0], 1) + global_V3d(vl2[1], 1)) * 0.5;
					bool in_l1 = global_grid_l1 ? (global_grid_l1->classify(mx, my) >= 0) : (classify_point_solid_polygon(Eigen::Vector2d(mx,my), polygon_refs_l1, all_polygons, *global_filter_contours_l1) >= 0);
					bool in_l2 = global_grid_l2 ? (global_grid_l2->classify(mx, my) >= 0) : (classify_point_solid_polygon(Eigen::Vector2d(mx,my), polygon_refs_l2, all_polygons, *global_filter_contours_l2) >= 0);
					if (in_l1 && in_l2) rescue = true;
				}

				if (rescue) {
					is_valid[i] = true;
					global_t12_rescued_by_ima++;
				}
			}
		}
	}

	// Step 3: Count isolated removals and build face lists
	global_t12_removed_by_isolated = 0;
	for (size_t i = 0; i < n_t12; ++i) {
		if (survives_pre[i] && !is_valid[i])
			global_t12_removed_by_isolated++;
	}

	global_faces_t12_valid.clear();
	global_faces_t12_removed.clear();
	delaunay3d_T12_valid = 0;
	delaunay3d_T12_removed = 0;

	for (size_t i = 0; i < n_t12; ++i) {
		auto& target = is_valid[i] ? global_faces_t12_valid : global_faces_t12_removed;
		if (is_valid[i]) delaunay3d_T12_valid++;
		else delaunay3d_T12_removed++;

		for (int lf = 0; lf < 4; ++lf)
			target.push_back(global_t12_tets[i].faces[lf]);
	}

	delaunay3d_T12_valid_faces = static_cast<int>(global_faces_t12_valid.size());
	delaunay3d_T12_removed_faces = static_cast<int>(global_faces_t12_removed.size());
	delaunay3d_all_valid_faces = static_cast<int>(global_faces_t1_valid.size())
		+ static_cast<int>(global_faces_t2_valid.size())
		+ delaunay3d_T12_valid_faces;
}

// =======================
// OBJ -> CNT SLICING
// =======================
bool generate_cnt_from_obj(const std::string& obj_path, const std::string& cnt_path, int num_slices, bool scale_coords)
{
	Eigen::MatrixXd V;
	Eigen::MatrixXi F;
	if (!igl::readOBJ(obj_path, V, F) || F.rows() == 0) {
		std::cerr << "ERROR: Could not read OBJ: " << obj_path << "\n";
		return false;
	}
	std::cout << "OBJ loaded: " << V.rows() << " vertices, " << F.rows() << " faces\n";

	// Find Z range
	double z_min = V.col(2).minCoeff();
	double z_max = V.col(2).maxCoeff();
	double z_range = z_max - z_min;
	std::cout << "Z range: [" << z_min << ", " << z_max << "] (range=" << z_range << ")\n";

	if (z_range < 1e-10) {
		std::cerr << "ERROR: Mesh is flat in Z.\n";
		return false;
	}

	// Generate slice Z values (exclude exact min/max to avoid degenerate slices)
	// Base spacing between slices
	double margin = z_range / (num_slices + 1);
	std::vector<double> slice_z;

	// Small tolerance for vertex coincidence
	double tol_snap = 1e-12;

	// Epsilon base: fraction of inter-slice distance. We'll try to nudge
	// slices that fall exactly on vertex Z by +/- (margin/100) (and grow
	// if needed) instead of removing that slice.
	double eps_base = std::max(margin / 100.0, 1e-12);

	for (int i = 1; i <= num_slices; ++i) {
		double z = z_min + i * margin;
		double original_z = z;

		// If a slice lies exactly on a vertex Z (within tol), try to nudge it
		// slightly (prefer + direction) so we don't lose information.
		bool collision = false;
		for (int vi = 0; vi < (int)V.rows(); ++vi) {
			double vz = V(vi, 2);
			if (std::abs(vz - z) <= tol_snap) { collision = true; break; }
		}
		if (collision) {
			double eps = eps_base;
			bool moved = false;
			for (int attempt = 0; attempt < 40 && !moved; ++attempt) {
				double z_plus = original_z + eps;
				double z_minus = original_z - eps;
				// prefer moving upward if inside range
				if (z_plus < z_max - 1e-12) {
					bool ok = true;
					for (int vi = 0; vi < (int)V.rows(); ++vi) if (std::abs(V(vi,2) - z_plus) <= tol_snap) { ok = false; break; }
					if (ok) { z = z_plus; moved = true; break; }
				}
				if (z_minus > z_min + 1e-12) {
					bool ok = true;
					for (int vi = 0; vi < (int)V.rows(); ++vi) if (std::abs(V(vi,2) - z_minus) <= tol_snap) { ok = false; break; }
					if (ok) { z = z_minus; moved = true; break; }
				}
				eps *= 2.0; // increase and retry
			}
			if (moved) {
				std::cout << "  [SLICE ADJUST] slice " << i << " adjusted from " << original_z << " to " << z << " to avoid vertex plane\n";
			} else {
				std::cout << "  [SLICE ADJUST] slice " << i << " could not be adjusted (keeps " << original_z << ")\n";
			}
		}

		slice_z.push_back(z);
	}

	std::cout << "Generating " << num_slices << " slices...\n";

	// Tolerance for snapping endpoints to the same grid cell
	const double tol = 1e-8;
	using KeyT = std::pair<long long, long long>;
	auto make_key = [&](const Eigen::Vector2d& p) -> KeyT {
		return {static_cast<long long>(std::round(p.x() / tol)),
				static_cast<long long>(std::round(p.y() / tol))};
	};

	// First pass: collect all level data into memory so we can count levels
	struct SliceData {
		double z;
		std::vector<std::vector<Eigen::Vector2d>> contours;
	};
	std::vector<SliceData> all_slices;

	for (int si = 0; si < (int)slice_z.size(); ++si) {
		double z = slice_z[si];

		// Find all triangle-plane intersections at this Z
		struct Segment { Eigen::Vector2d a, b; };
		std::vector<Segment> segments;

		for (int fi = 0; fi < F.rows(); ++fi) {
			Eigen::Vector3d p0 = V.row(F(fi, 0));
			Eigen::Vector3d p1 = V.row(F(fi, 1));
			Eigen::Vector3d p2 = V.row(F(fi, 2));

			double d0 = p0.z() - z;
			double d1 = p1.z() - z;
			double d2 = p2.z() - z;

			int above = (d0 > 1e-12) + (d1 > 1e-12) + (d2 > 1e-12);
			int below = (d0 < -1e-12) + (d1 < -1e-12) + (d2 < -1e-12);

			if (above == 0 || below == 0) continue;

			Eigen::Vector3d pts[3] = {p0, p1, p2};
			double dists[3] = {d0, d1, d2};
			std::vector<Eigen::Vector2d> hits;

			for (int e = 0; e < 3 && hits.size() < 2; ++e) {
				int ea = e, eb = (e + 1) % 3;
				double da = dists[ea], db = dists[eb];

				if ((da > 1e-12 && db < -1e-12) || (da < -1e-12 && db > 1e-12)) {
					// Proper crossing: interpolate
					double t = da / (da - db);
					Eigen::Vector3d hit = pts[ea] + t * (pts[eb] - pts[ea]);
					hits.emplace_back(hit.x(), hit.y());
				}
				else if (std::abs(da) <= 1e-12 && std::abs(db) > 1e-12) {
					// Vertex ea is exactly on the plane, eb is strictly off.
					// For a manifold mesh the 1-ring of ea has exactly 2
					// crossing triangles, so ea will appear with degree 2 in
					// the adjacency graph (no T-junction). Using make_key
					// snapping ensures all triangles that share ea produce the
					// same key, so they chain correctly.
					hits.emplace_back(pts[ea].x(), pts[ea].y());
				}
				// Vertex eb exactly on plane is handled when that edge is
				// visited as ea in the next triangle that shares it, so we
				// skip the eb==0 case here to avoid duplicate hits within
				// the same triangle.
			}

			if (hits.size() == 2) {
				// Skip degenerate segments (both endpoints identical)
				if (make_key(hits[0]) != make_key(hits[1]))
					segments.push_back({hits[0], hits[1]});
			}
		}

		if (segments.empty()) continue;

		// Chain segments into closed contours using adjacency map (O(n) per contour)
		// Build adjacency: endpoint key -> list of (segment_index, which_end)
		struct SegEnd { size_t seg_idx; int end; }; // end: 0=a, 1=b
		std::map<KeyT, std::vector<SegEnd>> adj;
		for (size_t i = 0; i < segments.size(); ++i) {
			adj[make_key(segments[i].a)].push_back({i, 0});
			adj[make_key(segments[i].b)].push_back({i, 1});
		}

		std::vector<bool> used(segments.size(), false);
		std::vector<std::vector<Eigen::Vector2d>> contours;

		for (size_t start = 0; start < segments.size(); ++start) {
			if (used[start]) continue;

			// Build chain using deque for O(1) push_front/push_back
			std::deque<Eigen::Vector2d> chain;
			chain.push_back(segments[start].a);
			chain.push_back(segments[start].b);
			used[start] = true;

			// Grow forward: follow from tail
			bool growing = true;
			while (growing) {
				growing = false;
				auto tail_k = make_key(chain.back());
				auto it = adj.find(tail_k);
				if (it != adj.end()) {
					for (auto& se : it->second) {
						if (used[se.seg_idx]) continue;
						// Add the OTHER endpoint of this segment
						const auto& seg = segments[se.seg_idx];
						Eigen::Vector2d next_pt = (se.end == 0) ? seg.b : seg.a;
						chain.push_back(next_pt);
						used[se.seg_idx] = true;
						growing = true;
						break;
					}
				}
			}

			// Grow backward: follow from head
			growing = true;
			while (growing) {
				growing = false;
				auto head_k = make_key(chain.front());
				auto it = adj.find(head_k);
				if (it != adj.end()) {
					for (auto& se : it->second) {
						if (used[se.seg_idx]) continue;
						const auto& seg = segments[se.seg_idx];
						Eigen::Vector2d prev_pt = (se.end == 0) ? seg.b : seg.a;
						chain.push_front(prev_pt);
						used[se.seg_idx] = true;
						growing = true;
						break;
					}
				}
			}

			// Check if closed (first == last within tolerance) and remove duplicate
			if (chain.size() >= 3 && make_key(chain.front()) == make_key(chain.back())) {
				chain.pop_back();
			}

			if (chain.size() >= 3) {
				contours.emplace_back(chain.begin(), chain.end());
			}
		}

		if (contours.empty()) continue;

		// ---------------------------------------------------------------
		// VALIDATION: closed contours + no cross-contour intersections
		// ---------------------------------------------------------------
		{
			// Segment-segment intersection test (strict: interior crossing only)
			auto seg_intersect = [](
				double ax, double ay, double bx, double by,
				double cx, double cy, double dx, double dy) -> bool
			{
				double denom = (bx-ax)*(dy-cy) - (by-ay)*(dx-cx);
				if (std::abs(denom) < 1e-15) return false;
				double t = ((cx-ax)*(dy-cy) - (cy-ay)*(dx-cx)) / denom;
				double u = ((cx-ax)*(by-ay) - (cy-ay)*(bx-ax)) / denom;
				return (t > 1e-9 && t < 1.0-1e-9 && u > 1e-9 && u < 1.0-1e-9);
			};

			int n_contours = static_cast<int>(contours.size());

			// 1) Closure check
			for (int ci = 0; ci < n_contours; ++ci) {
				const auto& c = contours[ci];
				if (c.size() < 3) {
					std::cerr << "WARNING: slice z=" << std::fixed << std::setprecision(4) << z
						<< " contour " << ci << " has only " << c.size() << " points (degenerate)\n";
				}
				// The chaining already removes the duplicate closing point,
				// so the contour is implicitly closed (last segment goes back to first).
				// We just verify the first and last points are NOT identical (no leftover duplicate).
				if (c.size() >= 2 && make_key(c.front()) == make_key(c.back())) {
					std::cerr << "WARNING: slice z=" << z
						<< " contour " << ci << " still has duplicate first/last point\n";
				}
			}

			// 2) Cross-contour intersection check
			// Root cause: T-junctions in the mesh create branching points where the greedy
			// chaining arbitrarily splits one contour into two crossing chains.
			// Fix strategy: when two contours intersect, merge their segments and re-chain them.
			int intersection_count = 0;
			std::set<int> contours_to_merge; // indices of contours involved in intersections
			for (int ci = 0; ci < n_contours; ++ci) {
				const auto& A = contours[ci];
				int na = static_cast<int>(A.size());
				for (int cj = ci + 1; cj < n_contours; ++cj) {
					const auto& B = contours[cj];
					int nb = static_cast<int>(B.size());
					bool found = false;
					for (int ia = 0; ia < na && !found; ++ia) {
						int ia2 = (ia + 1) % na;
						for (int ib = 0; ib < nb && !found; ++ib) {
							int ib2 = (ib + 1) % nb;
							if (seg_intersect(
								A[ia].x(), A[ia].y(), A[ia2].x(), A[ia2].y(),
								B[ib].x(), B[ib].y(), B[ib2].x(), B[ib2].y()))
							{
								std::cerr << "WARNING: slice z=" << std::fixed << std::setprecision(4) << z
									<< " contour " << ci << " (size=" << na << ")"
									<< " intersects contour " << cj << " (size=" << nb << ")"
									<< " at seg " << ia << "-" << ia2
									<< " x seg " << ib << "-" << ib2 << "\n";
								contours_to_merge.insert(ci);
								contours_to_merge.insert(cj);
								intersection_count++;
								found = true;
							}
						}
					}
				}
			}

			// --- FIX: Re-chain intersecting contours by merging their segments ---
			if (!contours_to_merge.empty()) {
				std::cerr << "  [FIX] Attempting to re-chain " << contours_to_merge.size()
					<< " intersecting contour(s) at z=" << std::fixed << std::setprecision(4) << z << "\n";

				// Collect all segments from the conflicting contours
				std::vector<Segment> fix_segs;
				for (int ci : contours_to_merge) {
					const auto& c = contours[ci];
					int nc = static_cast<int>(c.size());
					for (int k = 0; k < nc; ++k)
						fix_segs.push_back({c[k], c[(k + 1) % nc]});
				}

				// Re-chain using the same adjacency approach but prioritizing
				// the path that keeps the contour non-self-intersecting
				std::map<KeyT, std::vector<SegEnd>> fix_adj;
				for (size_t i = 0; i < fix_segs.size(); ++i) {
					fix_adj[make_key(fix_segs[i].a)].push_back({i, 0});
					fix_adj[make_key(fix_segs[i].b)].push_back({i, 1});
				}

				std::vector<bool> fix_used(fix_segs.size(), false);
				std::vector<std::vector<Eigen::Vector2d>> fixed_contours;

				for (size_t start = 0; start < fix_segs.size(); ++start) {
					if (fix_used[start]) continue;

					std::deque<Eigen::Vector2d> chain;
					chain.push_back(fix_segs[start].a);
					chain.push_back(fix_segs[start].b);
					fix_used[start] = true;

					bool growing = true;
					while (growing) {
						growing = false;
						auto tail_k = make_key(chain.back());
						auto it = fix_adj.find(tail_k);
						if (it != fix_adj.end()) {
							for (auto& se : it->second) {
								if (fix_used[se.seg_idx]) continue;
								Eigen::Vector2d next_pt = (se.end == 0) ? fix_segs[se.seg_idx].b : fix_segs[se.seg_idx].a;
								chain.push_back(next_pt);
								fix_used[se.seg_idx] = true;
								growing = true;
								break;
							}
						}
					}
					growing = true;
					while (growing) {
						growing = false;
						auto head_k = make_key(chain.front());
						auto it = fix_adj.find(head_k);
						if (it != fix_adj.end()) {
							for (auto& se : it->second) {
								if (fix_used[se.seg_idx]) continue;
								Eigen::Vector2d prev_pt = (se.end == 0) ? fix_segs[se.seg_idx].b : fix_segs[se.seg_idx].a;
								chain.push_front(prev_pt);
								fix_used[se.seg_idx] = true;
								growing = true;
								break;
							}
						}
					}

					if (chain.size() >= 3 && make_key(chain.front()) == make_key(chain.back()))
						chain.pop_back();
					if (chain.size() >= 3)
						fixed_contours.emplace_back(chain.begin(), chain.end());
				}

				// Replace the intersecting contours with the re-chained ones
				std::vector<std::vector<Eigen::Vector2d>> new_contours;
				for (int ci = 0; ci < n_contours; ++ci) {
					if (contours_to_merge.count(ci) == 0)
						new_contours.push_back(contours[ci]);
				}
				for (auto& fc : fixed_contours)
					new_contours.push_back(std::move(fc));
				contours = std::move(new_contours);

				std::cerr << "  [FIX] Re-chaining produced " << fixed_contours.size()
					<< " contour(s) from " << contours_to_merge.size() << " intersecting ones\n";

				// Re-validate after fix
				int remaining = 0;
				int nc2 = static_cast<int>(contours.size());
				for (int ci = 0; ci < nc2; ++ci) {
					int na = static_cast<int>(contours[ci].size());
					for (int cj = ci + 1; cj < nc2; ++cj) {
						int nb = static_cast<int>(contours[cj].size());
						for (int ia = 0; ia < na; ++ia) {
							int ia2 = (ia + 1) % na;
							for (int ib = 0; ib < nb; ++ib) {
								int ib2 = (ib + 1) % nb;
								if (seg_intersect(
									contours[ci][ia].x(), contours[ci][ia].y(),
									contours[ci][ia2].x(), contours[ci][ia2].y(),
									contours[cj][ib].x(), contours[cj][ib].y(),
									contours[cj][ib2].x(), contours[cj][ib2].y()))
								{ remaining++; goto done_recheck; }
							}
						}
					}
				}
				done_recheck:
				if (remaining == 0)
					std::cerr << "  [FIX] Intersections resolved at z=" << z << "\n";
				else
					std::cerr << "  [FIX] " << remaining << " intersection(s) still remain at z=" << z
						<< " (T-junction in mesh may need mesh repair)\n";
			}

			if (intersection_count == 0 && n_contours > 1) {
				std::cout << "  [OK] slice z=" << std::fixed << std::setprecision(4) << z
					<< ": " << n_contours << " contours, all closed, no intersections\n";
			} else if (intersection_count > 0) {
				std::cerr << "  [!!] slice z=" << z
					<< ": " << intersection_count << " intersecting contour pair(s) found\n";
			}
		}
		// ---------------------------------------------------------------

		SliceData sd;
		sd.z = z;
		sd.contours = std::move(contours);
		all_slices.push_back(std::move(sd));

		int total_pts = 0;
		for (const auto& c : all_slices.back().contours) total_pts += static_cast<int>(c.size());
		std::cout << "  Slice " << (si + 1) << "/" << num_slices
			<< " z=" << std::fixed << std::setprecision(4) << z
			<< " segments=" << segments.size()
			<< " contours=" << all_slices.back().contours.size()
			<< " points=" << total_pts << "\n";
	}

	// Write CNT file with header
	std::ofstream out(cnt_path);
	if (!out.is_open()) {
		std::cerr << "ERROR: Could not open output file: " << cnt_path << "\n";
		return false;
	}

	// Compute scale based on smallest non-zero absolute coordinate value (if requested).
	double cnt_scale    = 1.0;
	double cnt_offset_x = 0.0;
	double cnt_offset_y = 0.0;

	if (scale_coords) {
		double min_abs_value = 1e30;
		double bb_min_x = 1e30, bb_min_y = 1e30, bb_max_x = -1e30, bb_max_y = -1e30;

		for (const auto& slice : all_slices) {
			for (const auto& c : slice.contours) {
				for (const auto& p : c) {
					if (p.x() < bb_min_x) bb_min_x = p.x();
					if (p.y() < bb_min_y) bb_min_y = p.y();
					if (p.x() > bb_max_x) bb_max_x = p.x();
					if (p.y() > bb_max_y) bb_max_y = p.y();
					double abs_x = std::abs(p.x());
					double abs_y = std::abs(p.y());
					if (abs_x > 1e-10 && abs_x < min_abs_value) min_abs_value = abs_x;
					if (abs_y > 1e-10 && abs_y < min_abs_value) min_abs_value = abs_y;
				}
			}
		}

		if (min_abs_value > 1e10) {
			double range_x = bb_max_x - bb_min_x;
			double range_y = bb_max_y - bb_min_y;
			min_abs_value = std::min(range_x, range_y);
		}

		if (min_abs_value > 1e-10)
			cnt_scale = 100.0 / min_abs_value;

		cnt_offset_x = bb_min_x;
		cnt_offset_y = bb_min_y;

		std::cout << "CNT scaling enabled:\n";
		std::cout << "  Scale factor: " << cnt_scale << "\n";
		std::cout << "  Offset: (" << cnt_offset_x << ", " << cnt_offset_y << ")\n";
	} else {
		std::cout << "CNT scaling disabled: writing raw coordinates.\n";
	}

	int num_levels = static_cast<int>(all_slices.size());
	out << std::fixed;
	// Write only the level count in the S header (no scale/offset numbers).
	out << "S " << num_levels << "\n";

	int total_contours = 0;
	for (const auto& slice : all_slices) {
		int total_verts = 0;
		for (const auto& c : slice.contours) total_verts += static_cast<int>(c.size());

        out << "v " << total_verts << " z " << std::setprecision(10) << slice.z << "\n";
		for (const auto& c : slice.contours) {
			out << "{\n";
			for (const auto& p : c) {
				double sx = (p.x() - cnt_offset_x) * cnt_scale;
				double sy = (p.y() - cnt_offset_y) * cnt_scale;
				out << std::fixed << std::setprecision(10) << sx << " " << sy << "\n";
			}
			out << "}\n";
			total_contours++;
		}
	}

	out.close();
	std::cout << "\nCNT file written: " << cnt_path << "\n";
	std::cout << "Levels: " << num_levels
		<< ", total contours: " << total_contours << "\n";
	return true;
}

// ---------------------------------------------------------------------------------------//
int main() {
	GEO::initialize(GEO::GEOGRAM_INSTALL_ALL);

	// =======================
	// OBJ -> CNT GENERATION OPTION
	// =======================
	std::string filename;
	{
		char gen_choice;
		std::cout << "Do you want to generate a CNT file from an OBJ? (G for Generate, S for Skip): ";
		std::cin >> gen_choice;

		if (gen_choice != 'G' && gen_choice != 'g') {
			const std::string data_folder = "C:/Users/DELL/Desktop/Data/";
			std::vector<std::string> cnt_files;
			for (const auto& entry : std::filesystem::directory_iterator(data_folder)) {
				if (entry.is_regular_file() && entry.path().extension() == ".cnt")
					cnt_files.push_back(entry.path().filename().string());
			}
			std::sort(cnt_files.begin(), cnt_files.end());

			if (cnt_files.empty()) {
				std::cerr << "No .cnt files found in " << data_folder << "\n";
				return -1;
			}

			std::cout << "\nCNT files found in " << data_folder << ":\n";
			for (size_t i = 0; i < cnt_files.size(); ++i)
				std::cout << "  " << (i + 1) << ". " << cnt_files[i] << "\n";

			std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
			std::cout << "Enter CNT filename (or number, Enter = select 1): ";
			std::string cnt_input;
			std::getline(std::cin, cnt_input);

			if (cnt_input.empty()) {
				filename = data_folder + cnt_files[0];
			} else {
				bool is_number = !cnt_input.empty();
				for (char c : cnt_input) if (!std::isdigit(c)) { is_number = false; break; }
				if (is_number) {
					int idx = std::stoi(cnt_input) - 1;
					if (idx >= 0 && idx < (int)cnt_files.size())
						filename = data_folder + cnt_files[idx];
					else {
						std::cerr << "Invalid number.\n";
						return -1;
					}
				} else {
					filename = data_folder + cnt_input;
				}
			}
			std::cout << "Using: " << filename << "\n";
		}

		if (gen_choice == 'G' || gen_choice == 'g') {
			const std::string data_folder = "C:/Users/DELL/Desktop/Data/";

			// List available OBJ files in the data folder
			std::cout << "\nOBJ files found in " << data_folder << ":\n";
			int obj_count = 0;
			for (const auto& entry : std::filesystem::directory_iterator(data_folder)) {
				if (entry.is_regular_file() && entry.path().extension() == ".obj") {
					std::cout << "  - " << entry.path().filename().string() << "\n";
					obj_count++;
				}
			}
			if (obj_count == 0)
				std::cout << "  (none found)\n";

			std::string obj_name, cnt_name;
			int num_slices;

			std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
			std::cout << "\nEnter OBJ filename (e.g. model.obj): ";
			std::getline(std::cin, obj_name);

			std::cout << "Enter output CNT filename (e.g. model.cnt): ";
			std::getline(std::cin, cnt_name);

			std::cout << "Enter number of slices: ";
			std::cin >> num_slices;

			if (num_slices < 2) {
				std::cerr << "Need at least 2 slices.\n";
				return -1;
			}

			char scale_choice;
			std::cout << "Scale coordinates? (Y for Yes, N for No): ";
			std::cin >> scale_choice;
			bool do_scale = (scale_choice == 'Y' || scale_choice == 'y');

			std::string obj_input = data_folder + obj_name;
			std::string cnt_output = data_folder + cnt_name;

			if (!generate_cnt_from_obj(obj_input, cnt_output, num_slices, do_scale)) {
				std::cerr << "CNT generation failed.\n";
				return -1;
			}

			std::cout << "\nCNT generation complete. Showing contour preview...\n";

			// Read back the generated CNT to preview contours
			{
				std::ifstream preview_file(cnt_output);
				if (!preview_file.is_open()) {
					std::cerr << "Could not open generated CNT for preview: " << cnt_output << "\n";
					return -1;
				}

				std::vector<level_info> preview_levels;
				level_info preview_current;
				std::string preview_line;
				double pv_scale = 1.0, pv_off_x = 0.0, pv_off_y = 0.0;

				while (std::getline(preview_file, preview_line)) {
					if (preview_line.empty()) continue;
					if (preview_line[0] == 'S' || preview_line[0] == 's') {
						int dummy_levels;
						if (sscanf(preview_line.c_str(), "S %d %lf %lf %lf", &dummy_levels, &pv_scale, &pv_off_x, &pv_off_y) < 4)
							{ pv_scale = 1.0; pv_off_x = 0.0; pv_off_y = 0.0; }
						continue;
					}
					if (preview_line[0] == 'v') {
						if (!preview_current.contours.empty()) {
							preview_current.num_contours = static_cast<int>(preview_current.contours.size());
							preview_levels.push_back(preview_current);
						}
						preview_current = level_info();
						double z; int dv;
						sscanf(preview_line.c_str(), "v %d z %lf", &dv, &z);
						preview_current.zCoord = z;
					} else if (preview_line[0] == '{') {
						datapoints poly;
						while (std::getline(preview_file, preview_line) && preview_line[0] != '}') {
							double x, y;
							if (sscanf(preview_line.c_str(), "%lf %lf", &x, &y) == 2) {
								double rx = (pv_scale > 1e-15) ? x / pv_scale + pv_off_x : x;
								double ry = (pv_scale > 1e-15) ? y / pv_scale + pv_off_y : y;
								poly.points.emplace_back(rx, ry, preview_current.zCoord);
							}
						}
						if (!poly.points.empty())
							preview_current.contours.push_back(poly);
					}
				}
				if (!preview_current.contours.empty()) {
					preview_current.num_contours = static_cast<int>(preview_current.contours.size());
					preview_levels.push_back(preview_current);
				}
				preview_file.close();

				std::cout << "Preview: " << preview_levels.size() << " levels loaded\n";

				// Build viewer to show all contours
				igl::opengl::glfw::Viewer preview_viewer;
				igl::opengl::glfw::imgui::ImGuiPlugin preview_plugin;
				igl::opengl::glfw::imgui::ImGuiMenu preview_menu;
				preview_viewer.plugins.push_back(&preview_plugin);
				preview_plugin.widgets.push_back(&preview_menu);
				preview_viewer.core().background_color << 1.0, 1.0, 1.0, 1.0;
				preview_viewer.data().clear();

				// Bounding box
				double pMinX = 1e30, pMinY = 1e30, pMaxX = -1e30, pMaxY = -1e30;
				for (const auto& lv : preview_levels)
					for (const auto& c : lv.contours)
						for (const auto& p : c.points) {
							pMinX = std::min(pMinX, p.x()); pMinY = std::min(pMinY, p.y());
							pMaxX = std::max(pMaxX, p.x()); pMaxY = std::max(pMaxY, p.y());
						}
				Eigen::MatrixXd Vbox_p(2, 3); Eigen::MatrixXi Fbox_p(0, 3);
				Vbox_p.row(0) << pMinX, pMinY, 0;
				Vbox_p.row(1) << pMaxX, pMaxY, 0;

				// Draw all contours in black
				int total_levels = static_cast<int>(preview_levels.size());
				Eigen::RowVector3d black(0, 0, 0);
				for (int li = 0; li < total_levels; ++li) {
					for (const auto& contour : preview_levels[li].contours) {
						int n = static_cast<int>(contour.points.size());
						if (n == 0) continue;
						Eigen::MatrixXd V(n, 3);
						for (int j = 0; j < n; ++j) V.row(j) = contour.points[j];
						for (int j = 0; j < n; ++j)
							preview_viewer.data().add_edges(V.row(j), V.row((j + 1) % n), black);
					}
				}

				preview_viewer.data().set_mesh(Vbox_p, Fbox_p);
				preview_viewer.data().show_faces = false;

				int preview_total_contours = 0;
				for (const auto& lv : preview_levels)
					preview_total_contours += static_cast<int>(lv.contours.size());

				preview_menu.callback_draw_viewer_window = [&]() {
					ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(1, 1, 1, 1));
					ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0, 0, 0, 1));
					ImGui::PushStyleColor(ImGuiCol_TitleBg, ImVec4(0.85f, 0.85f, 0.85f, 1));
					ImGui::PushStyleColor(ImGuiCol_TitleBgActive, ImVec4(0.75f, 0.75f, 0.75f, 1));
					ImGui::Begin("CNT Preview");
					ImGui::SetWindowFontScale(1.4f);
					ImGui::Text("Levels: %d", total_levels);
					ImGui::Text("Total contours: %d", preview_total_contours);
					ImGui::SetWindowFontScale(1.0f);
					ImGui::End();
					ImGui::PopStyleColor(4);
				};

				std::cout << "Close the preview window to continue...\n";
				preview_viewer.launch();
			}

			char continue_choice;
			std::cout << "Continue with reconstruction using this CNT? (Y/N): ";
			std::cin >> continue_choice;
			if (continue_choice != 'Y' && continue_choice != 'y')
				return 0;

			filename = cnt_output;
		}
	}

	std::ifstream file(filename);

	if (!file.is_open()) {
		std::cerr << "Could not open file: " << filename << std::endl;
		return -1;
	}

	std::string line;

	std::vector<level_info> levels;
	level_info currentLevel;
	double cnt_scale = 1.0, cnt_off_x = 0.0, cnt_off_y = 0.0;

	while (std::getline(file, line))
	{
		if (line.empty())
			continue;

		// Parse header line: "S <levels> [<scale> <offset_x> <offset_y>]"
		if (line[0] == 'S' || line[0] == 's')
		{
			int dummy_levels;
			if (sscanf(line.c_str(), "S %d %lf %lf %lf", &dummy_levels, &cnt_scale, &cnt_off_x, &cnt_off_y) < 4)
				{ cnt_scale = 1.0; cnt_off_x = 0.0; cnt_off_y = 0.0; }
			else
				std::cout << "CNT scale: " << cnt_scale << " offset: (" << cnt_off_x << ", " << cnt_off_y << ")\n";
			continue;
		}

		if (line[0] == 'v')
		{
			if (!currentLevel.contours.empty()) {
				currentLevel.num_contours =
					static_cast<int>(currentLevel.contours.size());

				levels.push_back(currentLevel);
			}

			currentLevel = level_info();

			double z;
			int dummyVertices;

			if (sscanf(line.c_str(), "v %d z %lf", &dummyVertices, &z) == 2) {
				currentLevel.zCoord = z;
			}
		}

		else if (line[0] == '{')
		{
			datapoints poly;

			while (std::getline(file, line) && line[0] != '}')
			{
				double x, y;
				if (sscanf(line.c_str(), "%lf %lf", &x, &y) == 2)
				{
					double rx = (cnt_scale > 1e-15) ? x / cnt_scale + cnt_off_x : x;
					double ry = (cnt_scale > 1e-15) ? y / cnt_scale + cnt_off_y : y;
					poly.points.emplace_back(rx, ry, currentLevel.zCoord);
				}
			}

			if (!poly.points.empty())
			{
				currentLevel.contours.push_back(poly);
			}
		}
	}

	if (!currentLevel.contours.empty())
	{
		currentLevel.num_contours =
			static_cast<int>(currentLevel.contours.size());

		levels.push_back(currentLevel);
	}

	file.close();

	// Precompute bounding boxes for fast point-in-polygon rejection
	for (auto& level : levels)
		for (auto& contour : level.contours)
			contour.compute_bbox();

	int numSections = levels.size();
	std::cout << "Cross-sections found: " << numSections << std::endl;

	int startIdx, endIdx;
	int recon_algo = 2; // 1=NUAGES, 2=NUAGES_AUGMENTED, 3=2D_SHAPE_SIMILARITY
	char choice;

	std::cout << "Do you want to visualize all sections? (Y for Yes, N for No): ";
	std::cin >> choice;

	int resampleFactor = 1;
	std::vector<datapoints> resampled_l1;
	std::vector<datapoints> resampled_l2;

	if (choice == 'N' || choice == 'n') {

		std::cout << "Enter the section number to visualize (1 to "
			<< numSections << "): ";

		std::cin >> startIdx;

		endIdx = startIdx;

		if (startIdx < 1 || startIdx > numSections) {
			std::cerr << "Invalid selection. Exiting." << std::endl;
			return -1;
		}

		if (startIdx >= numSections) {
			std::cerr << "No existe nivel l2 para la secciÃƒÂ³n seleccionada.\n";
			return -1;
		}

		// ================================
		// Resample de puntos
		// ================================
		std::cout << "Enter resample factor (1 = original, 2 = add midpoints, etc.): ";
		std::cin >> resampleFactor;
		if (resampleFactor < 1) resampleFactor = 1;

		std::cout << "Reconstruction algorithm:\n";
		std::cout << "  1. NUAGES              (EMA augmentation only, no IMA)\n";
		std::cout << "  2. NUAGES AUGMENTED    (EMA + IMA augmentation)\n";
		std::cout << "  3. 2D Shape Similarity (per-group Delaunay)\n";
		std::cout << "  4. Simplified 2D Sim.  (NUAGES++ + group affinity filter)\n";
		std::cout << "Select (1-4): ";
		std::cin >> recon_algo;
		if (recon_algo < 1 || recon_algo > 4) recon_algo = 2;
		std::cout << "Algorithm: " << (recon_algo == 1 ? "NUAGES" : recon_algo == 3 ? "2D SHAPE SIMILARITY" : recon_algo == 4 ? "SIMPLIFIED 2D SHAPE SIMILARITY" : "NUAGES AUGMENTED") << "\n";

		resampled_l1 = resample_contours(levels[startIdx - 1].contours, resampleFactor);
		resampled_l2 = resample_contours(levels[startIdx].contours, resampleFactor);

		std::cout << "Resample factor: " << resampleFactor << std::endl;
		if (resampleFactor > 1) {
			for (size_t c = 0; c < resampled_l1.size(); ++c)
				std::cout << "  L1 contour " << c << ": " << levels[startIdx - 1].contours[c].points.size()
					<< " -> " << resampled_l1[c].points.size() << " points\n";
			for (size_t c = 0; c < resampled_l2.size(); ++c)
				std::cout << "  L2 contour " << c << ": " << levels[startIdx].contours[c].points.size()
					<< " -> " << resampled_l2[c].points.size() << " points\n";
		}

		// ================================
		// ConstrucciÃƒÂ³n del bosque nivel l1
		// ================================

		const auto& contours_l1 = levels[startIdx - 1].contours;
		int n_l1 = static_cast<int>(contours_l1.size());

		forest_l1.clear();
		forest_l1.resize(n_l1);

		for (int i = 0; i < n_l1; ++i) {
			forest_l1[i].contourId = i;
			forest_l1[i].parent = -1;
			forest_l1[i].children.clear();
			forest_l1[i].depth = 0;
		}

		for (int i = 0; i < n_l1; ++i) {

			const auto& testContour = contours_l1[i].points;

			if (testContour.empty())
				continue;

			auto testPoint = testContour[0];

			double minArea = std::numeric_limits<double>::infinity();
			int parent = -1;

			for (int j = 0; j < n_l1; ++j) {

				if (i == j) continue;

				if (pointInPolygon(testPoint, contours_l1[j].points)) {

					double area = polygonArea(contours_l1[j].points);

					if (area < minArea) {
						minArea = area;
						parent = j;
					}
				}
			}

			forest_l1[i].parent = parent;
		}

		for (int i = 0; i < n_l1; ++i) {
			if (forest_l1[i].parent != -1) {
				forest_l1[forest_l1[i].parent].children.push_back(i);
			}
		}

		for (int i = 0; i < n_l1; ++i) {
			forest_l1[i].depth = contourDepth(i, forest_l1);
		}

		// Ã¢Å“â€¦ GENERAR TEXTO DEL BOSQUE L1
		global_forest_text_l1 = format_forest_text(forest_l1, true);

		// ================================
		// ConstrucciÃƒÂ³n del bosque nivel l2
		// ================================

		const auto& contours_l2 = levels[startIdx].contours;
		int n_l2 = static_cast<int>(contours_l2.size());

		forest_l2.clear();
		forest_l2.resize(n_l2);

		for (int i = 0; i < n_l2; ++i) {
			forest_l2[i].contourId = i;
			forest_l2[i].parent = -1;
			forest_l2[i].children.clear();
			forest_l2[i].depth = 0;
		}

		for (int i = 0; i < n_l2; ++i) {

			const auto& testContour = contours_l2[i].points;

			if (testContour.empty())
				continue;

			auto testPoint = testContour[0];

			double minArea = std::numeric_limits<double>::infinity();
			int parent = -1;

			for (int j = 0; j < n_l2; ++j) {

				if (i == j) continue;

				if (pointInPolygon(testPoint, contours_l2[j].points)) {

					double area = polygonArea(contours_l2[j].points);

					if (area < minArea) {
						minArea = area;
						parent = j;
					}
				}
			}

			forest_l2[i].parent = parent;
		}

		for (int i = 0; i < n_l2; ++i) {
			if (forest_l2[i].parent != -1) {
				forest_l2[forest_l2[i].parent].children.push_back(i);
			}
		}

		for (int i = 0; i < n_l2; ++i) {
			forest_l2[i].depth = contourDepth(i, forest_l2);
		}

		// Ã¢Å“â€¦ GENERAR TEXTO DEL BOSQUE L2
		global_forest_text_l2 = format_forest_text(forest_l2, false);

		// ================================
		// ConstrucciÃƒÂ³n de arreglo global y referencias
		// ================================

		all_polygons.clear();
		polygon_refs_l1.clear();
		polygon_refs_l2.clear();

		// Agregar polÃƒÂ­gonos de L1
		for (int i = 0; i < n_l1; ++i) {
			int d = forest_l1[i].depth;

			if (d % 2 == 0) {
				solidpolygon P;
				P.contours.push_back(forest_l1[i].contourId);

				for (int c : forest_l1[i].children) {
					if (forest_l1[c].depth == d + 1) {
						P.contours.push_back(forest_l1[c].contourId);
					}
				}

				int idx = all_polygons.size();
				all_polygons.push_back(P);
				polygon_refs_l1.push_back(polygon_reference(idx, 0));
			}
		}

		// Agregar polÃƒÂ­gonos de L2
		for (int i = 0; i < n_l2; ++i) {
			int d = forest_l2[i].depth;

			if (d % 2 == 0) {
				solidpolygon P;
				P.contours.push_back(forest_l2[i].contourId);

				for (int c : forest_l2[i].children) {
					if (forest_l2[c].depth == d + 1) {
						P.contours.push_back(forest_l2[c].contourId);
					}
				}

				int idx = all_polygons.size();
				all_polygons.push_back(P);
				polygon_refs_l2.push_back(polygon_reference(idx, 1));
			}
		}

		// ================================
		// ConstrucciÃƒÂ³n de polÃƒÂ­gonos HOLE
		// ================================
		all_hole_polygons.clear();
		hole_refs_l1.clear();
		hole_refs_l2.clear();

		// Hole polygons de L1 (depth impar)
		for (int i = 0; i < n_l1; ++i) {
			int d = forest_l1[i].depth;

			if (d % 2 == 1) {
				solidpolygon H;
				H.contours.push_back(forest_l1[i].contourId);

				for (int c : forest_l1[i].children) {
					if (forest_l1[c].depth == d + 1) {
						H.contours.push_back(forest_l1[c].contourId);
					}
				}

				int idx = all_hole_polygons.size();
				all_hole_polygons.push_back(H);
				hole_refs_l1.push_back(polygon_reference(idx, 0));
			}
		}

		// Hole polygons de L2 (depth impar)
		for (int i = 0; i < n_l2; ++i) {
			int d = forest_l2[i].depth;

			if (d % 2 == 1) {
				solidpolygon H;
				H.contours.push_back(forest_l2[i].contourId);

				for (int c : forest_l2[i].children) {
					if (forest_l2[c].depth == d + 1) {
						H.contours.push_back(forest_l2[c].contourId);
					}
				}

				int idx = all_hole_polygons.size();
				all_hole_polygons.push_back(H);
				hole_refs_l2.push_back(polygon_reference(idx, 1));
			}
		}

		// =======================
		// BOSQUE NIVEL l1
		// =======================
		std::cout << "\nForest structure (Section " << startIdx << " - l1):\n";

		for (size_t i = 0; i < forest_l1.size(); ++i) {
			if (forest_l1[i].parent == -1) {
				printTree(forest_l1, i);
			}
		}

		// =========================
		// BOSQUE NIVEL l2
		// =========================
		std::cout << "\nForest structure (Section " << startIdx + 1 << " - l2):\n";

		for (size_t i = 0; i < forest_l2.size(); ++i) {
			if (forest_l2[i].parent == -1) {
				printTree(forest_l2, i);
			}
		}

		// ========================= POLÃƒÂGONOS L1
		std::cout << "\nPoligonos solidos detectados (l1):\n";

		for (size_t p = 0; p < polygon_refs_l1.size(); ++p) {
		 std::cout << "Poligono " << p + 1 << ": " 
					  << format_polygon_name_l1(&polygon_refs_l1[p], all_polygons, forest_l1) << "\n";
		}

		std::cout << "\nPoligonos solidos detectados (l2):\n";

		for (size_t p = 0; p < polygon_refs_l2.size(); ++p) {
		 std::cout << "Poligono " << p + 1 << ": " 
					  << format_polygon_name_l2(&polygon_refs_l2[p], all_polygons, forest_l2) << "\n";
		}

		// ========================= MATRIZ DE SIMILITUD
		std::cout << "\n=================================\n";
		std::cout << "SIMILARITY MATRIX\n";
		std::cout << "=================================\n\n";

		global_similarity_matrix = compute_similarity_matrix(
			polygon_refs_l1,
			polygon_refs_l2,
			all_polygons,
			contours_l1,
			contours_l2
		);

		for (const auto& entry : global_similarity_matrix) {
			std::string name_l1 = format_polygon_name_l1(entry.poly_l1_ref, all_polygons, forest_l1);
			std::string name_l2 = format_polygon_name_l2(entry.poly_l2_ref, all_polygons, forest_l2);

			std::cout << std::setw(12) << name_l1 << " vs " << std::setw(12) << name_l2 << " = ";

			if (entry.similarity_index < 1e-10) {
				std::cout << "No hay interseccion\n";
			}
			else {
			 std::cout << std::fixed << std::setprecision(4) << entry.similarity_index << "\n";
			}
		}

		std::cout << "\n=================================\n";

		// ========================= MAPPING GROUPS
		std::cout << "\n=================================\n";
		std::cout << "COMPUTING MAPPING GROUPS...\n";
		std::cout << "=================================\n";

		double threshold = 0.02;

		// Extraer polÃƒÂ­gonos individuales para compute_mapping_groups
		std::vector<solidpolygon> polygons_l1_only;
		std::vector<solidpolygon> polygons_l2_only;

		for (const auto& ref : polygon_refs_l1) {
			polygons_l1_only.push_back(all_polygons[ref.polygon_id]);
		}

		for (const auto& ref : polygon_refs_l2) {
			polygons_l2_only.push_back(all_polygons[ref.polygon_id]);
		}

		global_mapping_groups = compute_mapping_groups(
			polygons_l1_only, contours_l1,
			polygons_l2_only, contours_l2,
			threshold
		);

		std::cout << "\nMapping Groups found: " << global_mapping_groups.size() << "\n\n";

		for (size_t g = 0; g < global_mapping_groups.size(); ++g) {
			std::cout << "Group " << g + 1 << ":\n";

			std::cout << "  Level l1:  { ";
			if (global_mapping_groups[g].polygons_l1.empty()) {
				std::cout << "";
			}
			else {
				for (size_t i = 0; i < global_mapping_groups[g].polygons_l1.size(); ++i) {
					int pid = global_mapping_groups[g].polygons_l1[i];
					std::cout << format_polygon_name_l1(&polygon_refs_l1[pid], all_polygons, forest_l1);
					if (i + 1 < global_mapping_groups[g].polygons_l1.size())
						std::cout << ", ";
				}
			}
			std::cout << " }\n";

			std::cout << "  Level l2:  { ";
			if (global_mapping_groups[g].polygons_l2.empty()) {
				std::cout << "";
			}
			else {
				for (size_t i = 0; i < global_mapping_groups[g].polygons_l2.size(); ++i) {
					int pid = global_mapping_groups[g].polygons_l2[i];
					std::cout << format_polygon_name_l2(&polygon_refs_l2[pid], all_polygons, forest_l2);
					if (i + 1 < global_mapping_groups[g].polygons_l2.size())
						std::cout << ", ";
				}
			}
			std::cout << " }\n\n";
		}

		std::cout << "=================================\n";

		// ========================= COMPUTE HOLE MAPPING GROUPS + DEPURATED GROUPS (always)
		{
			std::vector<solidpolygon> hole_polys_l1_2d, hole_polys_l2_2d;
			for (const auto& ref : hole_refs_l1) hole_polys_l1_2d.push_back(all_hole_polygons[ref.polygon_id]);
			for (const auto& ref : hole_refs_l2) hole_polys_l2_2d.push_back(all_hole_polygons[ref.polygon_id]);
			global_hole_mapping_groups = compute_mapping_groups(hole_polys_l1_2d, contours_l1, hole_polys_l2_2d, contours_l2, threshold);

			auto fmt_poly_l1 = [&](const std::vector<int>& cids) -> std::string {
				if (cids.size() == 1) return std::string(1, 'A' + forest_l1[cids[0]].contourId);
				std::string s = "("; for (int cid : cids) s += char('A' + forest_l1[cid].contourId); return s + ")";
			};
			auto fmt_poly_l2 = [&](const std::vector<int>& cids) -> std::string {
				if (cids.size() == 1) return std::to_string(forest_l2[cids[0]].contourId + 1);
				std::string s = "("; for (int cid : cids) s += std::to_string(forest_l2[cid].contourId + 1); return s + ")";
			};

			// === LEVEL COMPUTATION ===
			struct GroupInfo2d { int idx; int level; bool is_empty; bool is_solid; };
			std::vector<GroupInfo2d> all_groups_2d;
			for (int gi = 0; gi < (int)global_mapping_groups.size(); ++gi) {
				const auto& g = global_mapping_groups[gi];
				bool empty = g.polygons_l1.empty() || g.polygons_l2.empty();
				int min_d = std::numeric_limits<int>::max();
				if (!empty) {
					for (int pid : g.polygons_l1) for (int cid : all_polygons[polygon_refs_l1[pid].polygon_id].contours) min_d = std::min(min_d, forest_l1[cid].depth);
					for (int pid : g.polygons_l2) for (int cid : all_polygons[polygon_refs_l2[pid].polygon_id].contours) min_d = std::min(min_d, forest_l2[cid].depth);
				}
				all_groups_2d.push_back({gi, min_d == std::numeric_limits<int>::max() ? -1 : min_d, empty, true});
			}
			for (int gi = 0; gi < (int)global_hole_mapping_groups.size(); ++gi) {
				const auto& g = global_hole_mapping_groups[gi];
				bool empty = g.polygons_l1.empty() || g.polygons_l2.empty();
				int min_d = std::numeric_limits<int>::max();
				if (!empty) {
					for (int pid : g.polygons_l1) for (int cid : hole_polys_l1_2d[pid].contours) min_d = std::min(min_d, forest_l1[cid].depth);
					for (int pid : g.polygons_l2) for (int cid : hole_polys_l2_2d[pid].contours) min_d = std::min(min_d, forest_l2[cid].depth);
				}
				all_groups_2d.push_back({gi, min_d == std::numeric_limits<int>::max() ? -1 : min_d, empty, false});
			}

			// === DEPURATION ===
			std::vector<int> sorted_2d;
			for (int i = 0; i < (int)all_groups_2d.size(); ++i) if (!all_groups_2d[i].is_empty) sorted_2d.push_back(i);
			// Solid groups before hole groups (within each category: deepest first).
			// This ensures solid polygons claim ALL their contours (outer + holes)
			// before hole groups can steal the shared inner contour IDs.
			// E.g. for A>B>C: solid (A,B) must claim B before hole group (B,C) does.
			std::stable_sort(sorted_2d.begin(), sorted_2d.end(), [&](int a, int b) {
				if (all_groups_2d[a].is_solid != all_groups_2d[b].is_solid)
					return all_groups_2d[a].is_solid > all_groups_2d[b].is_solid;
				return all_groups_2d[a].level > all_groups_2d[b].level;
			});
			std::set<int> used_l1_2d, used_l2_2d;
			global_final_groups_2d.clear();
			for (int sidx : sorted_2d) {
				const auto& ag = all_groups_2d[sidx];
				const mapping_group& gref = ag.is_solid ? global_mapping_groups[ag.idx] : global_hole_mapping_groups[ag.idx];
				std::vector<partial_polygon_entry> rem_l1, rem_l2;
				for (int pid : gref.polygons_l1) {
					const std::vector<int>& src = ag.is_solid ? all_polygons[polygon_refs_l1[pid].polygon_id].contours : hole_polys_l1_2d[pid].contours;
					partial_polygon_entry pp;
					for (int cid : src) if (!used_l1_2d.count(cid)) pp.contour_ids.push_back(cid);
					if (!pp.contour_ids.empty()) rem_l1.push_back(pp);
				}
				for (int pid : gref.polygons_l2) {
					const std::vector<int>& src = ag.is_solid ? all_polygons[polygon_refs_l2[pid].polygon_id].contours : hole_polys_l2_2d[pid].contours;
					partial_polygon_entry pp;
					for (int cid : src) if (!used_l2_2d.count(cid)) pp.contour_ids.push_back(cid);
					if (!pp.contour_ids.empty()) rem_l2.push_back(pp);
				}
				if (!rem_l1.empty() && !rem_l2.empty()) {
					final_mapping_group_2d fg;
					fg.l1_side = rem_l1; fg.l2_side = rem_l2;
					fg.level = ag.level; fg.is_solid = ag.is_solid;
					global_final_groups_2d.push_back(fg);
					for (const auto& pp : rem_l1) for (int cid : pp.contour_ids) used_l1_2d.insert(cid);
					for (const auto& pp : rem_l2) for (int cid : pp.contour_ids) used_l2_2d.insert(cid);
				}
			}

			// Print debug output when recon_algo == 3
			if (recon_algo == 3) {
				std::cout << "\n=================================\n";
				std::cout << "2D SHAPE SIMILARITY GROUPS\n";
				std::cout << "=================================\n";
				std::cout << "\nAll mapping groups with levels:\n";
				for (int i = 0; i < (int)all_groups_2d.size(); ++i) {
					const auto& ag = all_groups_2d[i];
					const mapping_group& gref = ag.is_solid ? global_mapping_groups[ag.idx] : global_hole_mapping_groups[ag.idx];
					std::string sl1 = "{", sl2 = "{";
					for (size_t pi = 0; pi < gref.polygons_l1.size(); ++pi) {
						if (pi) sl1 += ", ";
						const std::vector<int>& cids = ag.is_solid ? all_polygons[polygon_refs_l1[gref.polygons_l1[pi]].polygon_id].contours : hole_polys_l1_2d[gref.polygons_l1[pi]].contours;
						sl1 += fmt_poly_l1(cids);
					}
					for (size_t pi = 0; pi < gref.polygons_l2.size(); ++pi) {
						if (pi) sl2 += ", ";
						const std::vector<int>& cids = ag.is_solid ? all_polygons[polygon_refs_l2[gref.polygons_l2[pi]].polygon_id].contours : hole_polys_l2_2d[gref.polygons_l2[pi]].contours;
						sl2 += fmt_poly_l2(cids);
					}
					std::cout << "  " << (i+1) << ". [" << (ag.is_solid ? "solid" : "hole") << "] " << sl1 << "} vs " << sl2 << "}";
					std::cout << (ag.is_empty ? " -> EMPTY RELATION (skipped)\n" : (" -> Level = " + std::to_string(ag.level) + "\n"));
				}
				std::cout << "\nFinal depurated groups (" << global_final_groups_2d.size() << "):\n";
				for (int i = 0; i < (int)global_final_groups_2d.size(); ++i) {
					const auto& fg = global_final_groups_2d[i];
					std::cout << "  " << (i+1) << ". {";
					for (int pi = 0; pi < (int)fg.l1_side.size(); ++pi) { if (pi) std::cout << " "; std::cout << fmt_poly_l1(fg.l1_side[pi].contour_ids); }
					std::cout << "} vs {";
					for (int pi = 0; pi < (int)fg.l2_side.size(); ++pi) { if (pi) std::cout << " "; std::cout << fmt_poly_l2(fg.l2_side[pi].contour_ids); }
					std::cout << "} -> Level=" << fg.level << (fg.is_solid ? "" : " [hole]") << "\n";
				}
				std::cout << "=================================\n";
			}
		}

		// ========================= HOLE VS HOLE SIMILARITY

			// Helper: format a contour list as a polygon name for L1 (letters)
			auto fmt_poly_l1 = [&](const std::vector<int>& cids) -> std::string {
				if (cids.size() == 1) return std::string(1, 'A' + forest_l1[cids[0]].contourId);
				std::string s = "(";
				for (int cid : cids) s += char('A' + forest_l1[cid].contourId);
				return s + ")";
			};
			// Helper: format a contour list as a polygon name for L2 (numbers)
			auto fmt_poly_l2 = [&](const std::vector<int>& cids) -> std::string {
				if (cids.size() == 1) return std::to_string(forest_l2[cids[0]].contourId + 1);
				std::string s = "(";
				for (int cid : cids) s += std::to_string(forest_l2[cid].contourId + 1);
				return s + ")";
			};
		// ========================= HOLE VS HOLE SIMILARITY
		if (!hole_refs_l1.empty() && !hole_refs_l2.empty()) {
			std::cout << "\n=================================\n";
			std::cout << "HOLE vs HOLE SIMILARITY\n";
			std::cout << "=================================\n\n";

			hole_vs_hole_matrix.clear();
			for (size_t i = 0; i < hole_refs_l1.size(); ++i) {
				for (size_t j = 0; j < hole_refs_l2.size(); ++j) {
					const auto& poly_h1 = all_hole_polygons[hole_refs_l1[i].polygon_id];
					const auto& poly_h2 = all_hole_polygons[hole_refs_l2[j].polygon_id];

					double area_h1 = compute_polygon_area(poly_h1, contours_l1);
					double area_h2 = compute_polygon_area(poly_h2, contours_l2);

					polygon_intersection_result inter = compute_polygon_intersection_area(
						poly_h1, contours_l1, poly_h2, contours_l2);

					double similarity = 0.0;
					if (area_h1 > 1e-10 && area_h2 > 1e-10 && inter.intersection_area > 1e-10)
						similarity = std::min(inter.intersection_area / area_h1, inter.intersection_area / area_h2);

					similarity_matrix_entry entry;
					entry.poly_l1_ref = &hole_refs_l1[i];
					entry.poly_l2_ref = &hole_refs_l2[j];
					entry.similarity_index = similarity;
					entry.intersection_area = inter.intersection_area;
					entry.area_l1 = area_h1;
					entry.area_l2 = area_h2;
					entry.vertices = inter.vertices;
					entry.faces = inter.faces;
					hole_vs_hole_matrix.push_back(entry);
				}
			}

			for (const auto& entry : hole_vs_hole_matrix) {
				std::string n1 = format_polygon_name_l1(entry.poly_l1_ref, all_hole_polygons, forest_l1);
				std::string n2 = format_polygon_name_l2(entry.poly_l2_ref, all_hole_polygons, forest_l2);
				if (entry.intersection_area < 1e-10)
					std::cout << std::setw(12) << n1 << " vs " << std::setw(12) << n2 << " = No intersection\n";
				else {
					double r1 = entry.intersection_area / entry.area_l1;
					double r2 = entry.intersection_area / entry.area_l2;
					std::cout << std::setw(12) << n1 << " vs " << std::setw(12) << n2
						<< " = r1=" << std::fixed << std::setprecision(4) << r1
						<< "  r2=" << r2 << "\n";
				}
			}
			std::cout << "=================================\n";
		}

		// ========================= HOLE VS SOLID SIMILARITY
		if (!hole_refs_l1.empty() || !hole_refs_l2.empty()) {
			std::cout << "\n=================================\n";
			std::cout << "HOLE vs SOLID SIMILARITY\n";
			std::cout << "=================================\n\n";

			hole_vs_solid_matrix.clear();
			// Holes L1 vs Solids L2
			for (size_t i = 0; i < hole_refs_l1.size(); ++i) {
				for (size_t j = 0; j < polygon_refs_l2.size(); ++j) {
					const auto& poly_h = all_hole_polygons[hole_refs_l1[i].polygon_id];
					const auto& poly_s = all_polygons[polygon_refs_l2[j].polygon_id];

					double area_h = compute_polygon_area(poly_h, contours_l1);
					double area_s = compute_polygon_area(poly_s, contours_l2);

					polygon_intersection_result inter = compute_polygon_intersection_area(
						poly_h, contours_l1, poly_s, contours_l2);

					double similarity = 0.0;
					if (area_h > 1e-10 && area_s > 1e-10 && inter.intersection_area > 1e-10)
						similarity = std::min(inter.intersection_area / area_h, inter.intersection_area / area_s);

					similarity_matrix_entry entry;
					entry.poly_l1_ref = &hole_refs_l1[i];
					entry.poly_l2_ref = &polygon_refs_l2[j];
					entry.similarity_index = similarity;
					entry.intersection_area = inter.intersection_area;
					entry.area_l1 = area_h;
					entry.area_l2 = area_s;
					entry.vertices = inter.vertices;
					entry.faces = inter.faces;
					hole_vs_solid_matrix.push_back(entry);
				}
			}
			// Holes L2 vs Solids L1
			for (size_t i = 0; i < hole_refs_l2.size(); ++i) {
				for (size_t j = 0; j < polygon_refs_l1.size(); ++j) {
					const auto& poly_h = all_hole_polygons[hole_refs_l2[i].polygon_id];
					const auto& poly_s = all_polygons[polygon_refs_l1[j].polygon_id];

					double area_h = compute_polygon_area(poly_h, contours_l2);
					double area_s = compute_polygon_area(poly_s, contours_l1);

					polygon_intersection_result inter = compute_polygon_intersection_area(
						poly_h, contours_l2, poly_s, contours_l1);

					double similarity = 0.0;
					if (area_h > 1e-10 && area_s > 1e-10 && inter.intersection_area > 1e-10)
						similarity = std::min(inter.intersection_area / area_h, inter.intersection_area / area_s);

					similarity_matrix_entry entry;
					entry.poly_l1_ref = &hole_refs_l2[i];
					entry.poly_l2_ref = &polygon_refs_l1[j];
					entry.similarity_index = similarity;
					entry.intersection_area = inter.intersection_area;
					entry.area_l1 = area_h;
					entry.area_l2 = area_s;
					entry.vertices = inter.vertices;
					entry.faces = inter.faces;
					hole_vs_solid_matrix.push_back(entry);
				}
			}

			for (const auto& entry : hole_vs_solid_matrix) {
				std::string n1, n2;
				if (entry.poly_l1_ref->level == 0) {
					n1 = "H" + format_polygon_name_l1(entry.poly_l1_ref, all_hole_polygons, forest_l1);
					n2 = "S" + format_polygon_name_l2(entry.poly_l2_ref, all_polygons, forest_l2);
				} else {
					n1 = "H" + format_polygon_name_l2(entry.poly_l1_ref, all_hole_polygons, forest_l2);
					n2 = "S" + format_polygon_name_l1(entry.poly_l2_ref, all_polygons, forest_l1);
				}
				if (entry.intersection_area < 1e-10)
					std::cout << std::setw(14) << n1 << " vs " << std::setw(14) << n2 << " = No intersection\n";
				else {
					double r1 = entry.intersection_area / entry.area_l1;
					double r2 = entry.intersection_area / entry.area_l2;
					std::cout << std::setw(14) << n1 << " vs " << std::setw(14) << n2
						<< " = r1=" << std::fixed << std::setprecision(4) << r1
						<< "  r2=" << r2 << "\n";
				}
			}
			std::cout << "=================================\n";
		}
	}
	else {
		// Multi-section reconstruction
		int rangeStart, rangeEnd;
		std::cout << "Enter section range (e.g. 8-12): ";
		char dash;
		std::cin >> rangeStart >> dash >> rangeEnd;

		if (rangeStart < 1) rangeStart = 1;
		if (rangeEnd > numSections) rangeEnd = numSections;
		if (rangeStart >= rangeEnd) {
			std::cerr << "Invalid range. Need at least 2 sections.\n";
			return -1;
		}

		std::cout << "Enter resample factor (1 = original, 2 = add midpoints, etc.): ";
		std::cin >> resampleFactor;
		if (resampleFactor < 1) resampleFactor = 1;

		std::cout << "Reconstruction algorithm:\n";
		std::cout << "  1. NUAGES              (EMA augmentation only, no IMA)\n";
		std::cout << "  2. NUAGES AUGMENTED    (EMA + IMA augmentation)\n";
		std::cout << "  3. 2D Shape Similarity (not yet implemented)\n";
		std::cout << "Select (1-2): ";
		std::cin >> recon_algo;
		if (recon_algo < 1 || recon_algo > 2) recon_algo = 2;
		std::cout << "Algorithm: " << (recon_algo == 1 ? "NUAGES" : "NUAGES AUGMENTED") << "\n";

		startIdx = rangeStart;
		endIdx = rangeEnd;

		std::cout << "\n=================================\n";
		std::cout << "MULTI-SECTION RECONSTRUCTION\n";
		std::cout << "Range: " << rangeStart << " - " << rangeEnd << "\n";
		std::cout << "Pairs to process: " << (rangeEnd - rangeStart) << "\n";
		std::cout << "Resample factor: " << resampleFactor << "\n";
		std::cout << "=================================\n";

		// ================================================================
		// MULTI-SECTION RECONSTRUCTION - TWO-PASS APPROACH
		//
		// Pass 1: Compute IMA/EMA for all pairs first.
		// Then build the COMPLETE augmented point set for each level using
		// ALL neighboring-pair contributions BEFORE the Delaunay step.
		// Assign fixed global vertex offsets per level.
		//
		// Result: shared level Li has identical global vertex indices in
		// pair (Li-1,Li) AND pair (Li,Li+1).  The interior horizontal caps
		// at Li appear twice in the tetrahedra ? igl::boundary_facets
		// cancels them, leaving only the true outer skin.
		// ================================================================

		// ---- Per-pair data structure ----
		struct PairComputed {
			int idxL1, idxL2;
			std::vector<solidpolygon> polys;
			std::vector<polygon_reference> prefs_l1, prefs_l2;
			SolidRegionGrid grid_l1, grid_l2;
			medial_axes_result ma_l1, ma_l2;
			ema_projection_result proj_l1_on_l2; // EMA L1 solid verts landing in L2
			ema_projection_result proj_l2_on_l1; // EMA L2 solid verts landing in L1
		};

		std::vector<PairComputed> pair_computed;
		pair_computed.reserve(rangeEnd - rangeStart);

		// ---- Pass 1: forest, grids, Voronoi, IMA, EMA for every pair ----
		for (int pair = rangeStart; pair < rangeEnd; ++pair) {
			int idxL1 = pair - 1, idxL2 = pair;
			PairComputed pc;
			pc.idxL1 = idxL1; pc.idxL2 = idxL2;
			std::cout << "\n--- Pass1 pair " << pair << "-" << (pair+1)
				<< " (levels[" << idxL1 << "]-[" << idxL2 << "]) ---\n";

			// Forest + polygons L1
			const auto& contours_l1_pair = levels[idxL1].contours;
			int n_l1 = static_cast<int>(contours_l1_pair.size());
			std::vector<tree_node> forest_l1_pair(n_l1);
			for (int i = 0; i < n_l1; ++i) { forest_l1_pair[i].contourId=i; forest_l1_pair[i].parent=-1; forest_l1_pair[i].depth=0; }
			for (int i = 0; i < n_l1; ++i) {
				if (contours_l1_pair[i].points.empty()) continue;
				auto tp = contours_l1_pair[i].points[0];
				double minA = std::numeric_limits<double>::infinity(); int par = -1;
				for (int j = 0; j < n_l1; ++j) {
					if (i==j) continue;
					if (pointInPolygon(tp, contours_l1_pair[j].points)) {
						double a = polygonArea(contours_l1_pair[j].points);
						if (a < minA) { minA=a; par=j; }
					}
				}
				forest_l1_pair[i].parent = par;
			}
			for (int i=0;i<n_l1;++i) if (forest_l1_pair[i].parent!=-1) forest_l1_pair[forest_l1_pair[i].parent].children.push_back(i);
			for (int i=0;i<n_l1;++i) forest_l1_pair[i].depth = contourDepth(i, forest_l1_pair);
			for (int i=0;i<n_l1;++i) {
				if (forest_l1_pair[i].depth % 2 == 0) {
					solidpolygon P; P.contours.push_back(i);
					for (int c : forest_l1_pair[i].children) if (forest_l1_pair[c].depth==forest_l1_pair[i].depth+1) P.contours.push_back(c);
					int idx = static_cast<int>(pc.polys.size());
					pc.polys.push_back(P);
					pc.prefs_l1.push_back(polygon_reference(idx, 0));
				}
			}

			// Forest + polygons L2
			const auto& contours_l2_pair = levels[idxL2].contours;
			int n_l2 = static_cast<int>(contours_l2_pair.size());
			std::vector<tree_node> forest_l2_pair(n_l2);
			for (int i=0;i<n_l2;++i) { forest_l2_pair[i].contourId=i; forest_l2_pair[i].parent=-1; forest_l2_pair[i].depth=0; }
			for (int i=0;i<n_l2;++i) {
				if (contours_l2_pair[i].points.empty()) continue;
				auto tp = contours_l2_pair[i].points[0];
				double minA = std::numeric_limits<double>::infinity(); int par = -1;
				for (int j=0;j<n_l2;++j) {
					if (i==j) continue;
					if (pointInPolygon(tp, contours_l2_pair[j].points)) {
						double a = polygonArea(contours_l2_pair[j].points);
						if (a < minA) { minA=a; par=j; }
					}
				}
				forest_l2_pair[i].parent = par;
			}
			for (int i=0;i<n_l2;++i) if (forest_l2_pair[i].parent!=-1) forest_l2_pair[forest_l2_pair[i].parent].children.push_back(i);
			for (int i=0;i<n_l2;++i) forest_l2_pair[i].depth = contourDepth(i, forest_l2_pair);
			for (int i=0;i<n_l2;++i) {
				if (forest_l2_pair[i].depth % 2 == 0) {
					solidpolygon P; P.contours.push_back(i);
					for (int c : forest_l2_pair[i].children) if (forest_l2_pair[c].depth==forest_l2_pair[i].depth+1) P.contours.push_back(c);
					int idx = static_cast<int>(pc.polys.size());
					pc.polys.push_back(P);
					pc.prefs_l2.push_back(polygon_reference(idx, 1));
				}
			}

			// Grids
			pc.grid_l1.precompute(pc.prefs_l1, pc.polys, contours_l1_pair);
			pc.grid_l2.precompute(pc.prefs_l2, pc.polys, contours_l2_pair);

			// Voronoi + medial axes L1
			auto resampled_l1_pair = resample_contours(contours_l1_pair, resampleFactor);
			auto vor_l1 = compute_voronoi_edges(resampled_l1_pair);
			pc.ma_l1 = classify_voronoi_edges(vor_l1, pc.prefs_l1, pc.polys, contours_l1_pair, &pc.grid_l1);

			// Voronoi + medial axes L2
			auto resampled_l2_pair = resample_contours(contours_l2_pair, resampleFactor);
			auto vor_l2 = compute_voronoi_edges(resampled_l2_pair);
			pc.ma_l2 = classify_voronoi_edges(vor_l2, pc.prefs_l2, pc.polys, contours_l2_pair, &pc.grid_l2);

			// EMA projections
			pc.proj_l1_on_l2 = compute_ema_projection(pc.ma_l1.ema_edges, pc.prefs_l2, pc.polys, contours_l2_pair, &pc.grid_l2);
			pc.proj_l2_on_l1 = compute_ema_projection(pc.ma_l2.ema_edges, pc.prefs_l1, pc.polys, contours_l1_pair, &pc.grid_l1);

			std::cout << "  IMA_L1=" << pc.ma_l1.ima_edges.size()
				<< " IMA_L2=" << pc.ma_l2.ima_edges.size()
				<< " proj_L1->L2=" << pc.proj_l1_on_l2.solid_vertices.size()
				<< " proj_L2->L1=" << pc.proj_l2_on_l1.solid_vertices.size() << "\n";

			pair_computed.push_back(std::move(pc));
		}

		// Fix dangling pointers in grids: push_back(move) relocated the struct data,
		// so stored_poly_refs / stored_all_polys must be re-pointed to the in-vector addresses.
		for (auto& pc : pair_computed) {
			pc.grid_l1.stored_poly_refs = &pc.prefs_l1;
			pc.grid_l1.stored_all_polys = &pc.polys;
			pc.grid_l2.stored_poly_refs = &pc.prefs_l2;
			pc.grid_l2.stored_all_polys = &pc.polys;
		}

		// ---- Build complete augmented point sets for each level ----
		// Each level gets: contour pts + IMA (once) + EMA from ALL neighbors
		std::vector<std::vector<Eigen::Vector3d>> aug(numSections);
		std::vector<index_t> ima_start_lv(numSections, 0), ima_end_lv(numSections, 0);
		std::vector<bool> ima_done(numSections, false), contour_done(numSections, false);

		for (auto& pc : pair_computed) {
			int idxL1 = pc.idxL1, idxL2 = pc.idxL2;
			double z_l1 = levels[idxL1].zCoord, z_l2 = levels[idxL2].zCoord;

			if (!contour_done[idxL1]) {
				auto r = resample_contours(levels[idxL1].contours, resampleFactor);
				for (const auto& c : r) for (const auto& p : c.points) aug[idxL1].push_back(p);
				contour_done[idxL1] = true;
			}
			if (!contour_done[idxL2]) {
				auto r = resample_contours(levels[idxL2].contours, resampleFactor);
				for (const auto& c : r) for (const auto& p : c.points) aug[idxL2].push_back(p);
				contour_done[idxL2] = true;
			}
			if (!ima_done[idxL1]) {
				ima_start_lv[idxL1] = static_cast<index_t>(aug[idxL1].size());
				if (recon_algo >= 2) {
					for (const auto& e : pc.ma_l1.ima_edges) {
						aug[idxL1].emplace_back(e.first.x, e.first.y, z_l1);
						aug[idxL1].emplace_back(e.second.x, e.second.y, z_l1);
					}
				}
				ima_end_lv[idxL1] = static_cast<index_t>(aug[idxL1].size());
				ima_done[idxL1] = true;
			}
			if (!ima_done[idxL2]) {
				ima_start_lv[idxL2] = static_cast<index_t>(aug[idxL2].size());
				if (recon_algo >= 2) {
					for (const auto& e : pc.ma_l2.ima_edges) {
						aug[idxL2].emplace_back(e.first.x, e.first.y, z_l2);
						aug[idxL2].emplace_back(e.second.x, e.second.y, z_l2);
					}
				}
				ima_end_lv[idxL2] = static_cast<index_t>(aug[idxL2].size());
				ima_done[idxL2] = true;
			}
			// EMA neighbor contributions (each pair adds its cross-projection)
			for (const auto& v : pc.proj_l2_on_l1.solid_vertices) aug[idxL1].emplace_back(v.x(), v.y(), z_l1);
			for (const auto& v : pc.proj_l1_on_l2.solid_vertices) aug[idxL2].emplace_back(v.x(), v.y(), z_l2);
		}

		// ---- Assign fixed global vertex offsets per level ----
		std::vector<index_t> lv_offset(numSections, 0);
		index_t total_global_pts = 0;
		std::cout << "\n=== Augmented level sizes ===\n";
		for (int li = rangeStart - 1; li < rangeEnd; ++li) {
			lv_offset[li] = total_global_pts;
			total_global_pts += static_cast<index_t>(aug[li].size());
			std::cout << "  Level " << (li+1) << ": " << aug[li].size()
				<< " pts  global_offset=" << lv_offset[li] << "\n";
		}

		// Build global vertex matrix
		Eigen::MatrixXd V3d_multi(total_global_pts, 3);
		for (int li = rangeStart - 1; li < rangeEnd; ++li)
			for (size_t i = 0; i < aug[li].size(); ++i)
				V3d_multi.row(lv_offset[li] + i) << aug[li][i].x(), aug[li][i].y(), aug[li][i].z();

		// ---- Pass 2: Delaunay per pair, faces stored with GLOBAL indices ----
		// Shared-level caps cancel in boundary_facets (same global vertex indices).
		std::vector<std::array<index_t, 3>> accum_faces_t1_valid, accum_faces_t2_valid, accum_faces_t12_valid;
		std::vector<std::array<index_t, 4>> accum_tet_verts;

		// shared helpers for the isolated-filter hash set
		auto make_fk_p2 = [](index_t a, index_t b, index_t c) -> std::array<index_t,3> {
			std::array<index_t,3> f={a,b,c};
			if(f[0]>f[1]) std::swap(f[0],f[1]);
			if(f[1]>f[2]) std::swap(f[1],f[2]);
			if(f[0]>f[1]) std::swap(f[0],f[1]);
			return f;
		};
		struct FaceHashP2 { size_t operator()(const std::array<index_t,3>& a) const {
			size_t h=std::hash<index_t>{}(a[0]);
			h^=std::hash<index_t>{}(a[1])+0x9e3779b9+(h<<6)+(h>>2);
			h^=std::hash<index_t>{}(a[2])+0x9e3779b9+(h<<6)+(h>>2);
			return h;
		}};
		const int fvp2[4][3] = {{1,2,3},{0,2,3},{0,1,3},{0,1,2}};

		for (auto& pc : pair_computed) {
			int idxL1 = pc.idxL1, idxL2 = pc.idxL2;
			index_t n_l1_pts = static_cast<index_t>(aug[idxL1].size());
			index_t n_l2_pts = static_cast<index_t>(aug[idxL2].size());
			index_t N3d = n_l1_pts + n_l2_pts;

			// local index -> global index
			auto to_global = [&](index_t local) -> index_t {
				return local < n_l1_pts
					? lv_offset[idxL1] + local
					: lv_offset[idxL2] + (local - n_l1_pts);
			};

			// IMA global ranges for IMA rescue
			index_t g_ima_s_l1 = lv_offset[idxL1] + ima_start_lv[idxL1];
			index_t g_ima_e_l1 = lv_offset[idxL1] + ima_end_lv[idxL1];
			index_t g_ima_s_l2 = lv_offset[idxL2] + ima_start_lv[idxL2];
			index_t g_ima_e_l2 = lv_offset[idxL2] + ima_end_lv[idxL2];

			std::cout << "\n--- Pass2 pair " << (idxL1+1) << "-" << (idxL2+1) << " ---\n";
			std::cout << "  L1_pts=" << n_l1_pts << " L2_pts=" << n_l2_pts << "\n";

			// flat local coords: L1 first, then L2
			std::vector<double> coords_3d;
			coords_3d.reserve(N3d * 3);
			for (const auto& p2 : aug[idxL1]) { coords_3d.push_back(p2.x()); coords_3d.push_back(p2.y()); coords_3d.push_back(p2.z()); }
			for (const auto& p2 : aug[idxL2]) { coords_3d.push_back(p2.x()); coords_3d.push_back(p2.y()); coords_3d.push_back(p2.z()); }

			if (N3d < 4) { std::cout << "  Not enough pts, skipping.\n"; continue; }

			// accumulate results for this pair
			std::vector<std::array<index_t,3>> fv_t1, fv_t2, fv_t12;
			std::vector<std::array<index_t,4>> tv_t1, tv_t2, tv_t12;
			int cnt_t1v=0, cnt_t2v=0, cnt_t12v=0;

			// ============================================================
			// ALGO 3: 2D Shape Similarity per group, one Delaunay per pair
			// ============================================================
			if (recon_algo == 3) {
				const auto& contours_l1_pair = levels[idxL1].contours;
				const auto& contours_l2_pair = levels[idxL2].contours;
				int n_l1c = static_cast<int>(contours_l1_pair.size());
				int n_l2c = static_cast<int>(contours_l2_pair.size());

				// rebuild per-pair forests
				auto build_forest_pair = [&](std::vector<tree_node>& f, int n, const std::vector<datapoints>& cc){
					for(int i=0;i<n;++i){f[i].contourId=i;f[i].parent=-1;f[i].depth=0;}
					for(int i=0;i<n;++i){
						if(cc[i].points.empty()) continue;
						auto tp=cc[i].points[0]; double minA=1e30; int par=-1;
						for(int j=0;j<n;++j){ if(i==j) continue;
							if(pointInPolygon(tp,cc[j].points)){double a=polygonArea(cc[j].points);if(a<minA){minA=a;par=j;}}
						}
						f[i].parent=par;
					}
					for(int i=0;i<n;++i) if(f[i].parent!=-1) f[f[i].parent].children.push_back(i);
					for(int i=0;i<n;++i) f[i].depth=contourDepth(i,f);
				};
				std::vector<tree_node> fl1p(n_l1c), fl2p(n_l2c);
				build_forest_pair(fl1p,n_l1c,contours_l1_pair);
				build_forest_pair(fl2p,n_l2c,contours_l2_pair);

				// solid and hole polygons for this pair
				std::vector<solidpolygon> sp_l1,sp_l2,hp_l1,hp_l2;
				std::vector<polygon_reference> sr_l1,sr_l2;
				for(int i=0;i<n_l1c;++i){
					if(fl1p[i].depth%2==0){
						solidpolygon P; P.contours.push_back(i);
						for(int c:fl1p[i].children) if(fl1p[c].depth==fl1p[i].depth+1) P.contours.push_back(c);
						sr_l1.push_back(polygon_reference((int)sp_l1.size(),0)); sp_l1.push_back(P);
					} else { solidpolygon P; P.contours.push_back(i); hp_l1.push_back(P); }
				}
				for(int i=0;i<n_l2c;++i){
					if(fl2p[i].depth%2==0){
						solidpolygon P; P.contours.push_back(i);
						for(int c:fl2p[i].children) if(fl2p[c].depth==fl2p[i].depth+1) P.contours.push_back(c);
						sr_l2.push_back(polygon_reference((int)sp_l2.size(),1)); sp_l2.push_back(P);
					} else { solidpolygon P; P.contours.push_back(i); hp_l2.push_back(P); }
				}

				auto smg = compute_mapping_groups(sp_l1,contours_l1_pair,sp_l2,contours_l2_pair,0.02);
				auto hmg = compute_mapping_groups(hp_l1,contours_l1_pair,hp_l2,contours_l2_pair,0.02);

				// depuration: solid before hole, deepest first
				struct GI2p{ int idx; int level; bool is_empty; bool is_solid; };
				std::vector<GI2p> ag2;
				for(int gi=0;gi<(int)smg.size();++gi){
					const auto& g=smg[gi]; bool empty=g.polygons_l1.empty()||g.polygons_l2.empty();
					int md=std::numeric_limits<int>::max();
					if(!empty){
						for(int pid:g.polygons_l1) for(int cid:sp_l1[sr_l1[pid].polygon_id].contours) md=std::min(md,fl1p[cid].depth);
						for(int pid:g.polygons_l2) for(int cid:sp_l2[sr_l2[pid].polygon_id].contours) md=std::min(md,fl2p[cid].depth);
					}
					ag2.push_back({gi,md==std::numeric_limits<int>::max()?-1:md,empty,true});
				}
				for(int gi=0;gi<(int)hmg.size();++gi){
					const auto& g=hmg[gi]; bool empty=g.polygons_l1.empty()||g.polygons_l2.empty();
					int md=std::numeric_limits<int>::max();
					if(!empty){
						for(int pid:g.polygons_l1) for(int cid:hp_l1[pid].contours) md=std::min(md,fl1p[cid].depth);
						for(int pid:g.polygons_l2) for(int cid:hp_l2[pid].contours) md=std::min(md,fl2p[cid].depth);
					}
					ag2.push_back({gi,md==std::numeric_limits<int>::max()?-1:md,empty,false});
				}
				std::vector<int> srt2;
				for(int i=0;i<(int)ag2.size();++i) if(!ag2[i].is_empty) srt2.push_back(i);
				std::stable_sort(srt2.begin(),srt2.end(),[&](int a,int b){
					if(ag2[a].is_solid!=ag2[b].is_solid) return ag2[a].is_solid>ag2[b].is_solid;
					return ag2[a].level>ag2[b].level;
				});
				std::set<int> used1p,used2p;
				std::vector<final_mapping_group_2d> pair_fg;
				for(int sidx:srt2){
					const auto& ag=ag2[sidx];
					const mapping_group& gref=ag.is_solid?smg[ag.idx]:hmg[ag.idx];
					std::vector<partial_polygon_entry> rl1,rl2;
					for(int pid:gref.polygons_l1){
						const std::vector<int>& src2=ag.is_solid?sp_l1[sr_l1[pid].polygon_id].contours:hp_l1[pid].contours;
						partial_polygon_entry pp; for(int cid:src2) if(!used1p.count(cid)) pp.contour_ids.push_back(cid);
						if(!pp.contour_ids.empty()) rl1.push_back(pp);
					}
					for(int pid:gref.polygons_l2){
						const std::vector<int>& src2=ag.is_solid?sp_l2[sr_l2[pid].polygon_id].contours:hp_l2[pid].contours;
						partial_polygon_entry pp; for(int cid:src2) if(!used2p.count(cid)) pp.contour_ids.push_back(cid);
						if(!pp.contour_ids.empty()) rl2.push_back(pp);
					}
					if(!rl1.empty()&&!rl2.empty()){
						final_mapping_group_2d fg2; fg2.l1_side=rl1; fg2.l2_side=rl2; fg2.level=ag.level; fg2.is_solid=ag.is_solid;
						pair_fg.push_back(fg2);
						for(const auto& pp:rl1) for(int cid:pp.contour_ids) used1p.insert(cid);
						for(const auto& pp:rl2) for(int cid:pp.contour_ids) used2p.insert(cid);
					}
				}
				std::cout << "  Algo3 shape groups for pair: " << pair_fg.size() << "\n";

				// one Delaunay for all pair points, then filter per group
				Delaunay_var del3p = Delaunay::create(3, "BDEL");
				del3p->set_vertices(N3d, coords_3d.data());
				index_t nb_tp = del3p->nb_cells();
				std::cout << "  Tets=" << nb_tp << "\n";

				for (int gidx=0;gidx<(int)pair_fg.size();++gidx) {
					const auto& fg2 = pair_fg[gidx];
					std::cout << "    Group " << (gidx+1) << "/" << pair_fg.size() << "\n";

					std::vector<datapoints> gc_l1g,gc_l2g;
					for(const auto& pp:fg2.l1_side) for(int cid:pp.contour_ids) gc_l1g.push_back(contours_l1_pair[cid]);
					for(const auto& pp:fg2.l2_side) for(int cid:pp.contour_ids) gc_l2g.push_back(contours_l2_pair[cid]);
					if(gc_l1g.empty()||gc_l2g.empty()) continue;

					std::vector<solidpolygon> gp_l1g,gp_l2g;
					std::vector<polygon_reference> gr_l1g,gr_l2g;
					{int lc=0;
					 for(const auto& pp:fg2.l1_side){ solidpolygon P; for(int k=0;k<(int)pp.contour_ids.size();++k) P.contours.push_back(lc+k); gp_l1g.push_back(P); gr_l1g.push_back(polygon_reference((int)gp_l1g.size()-1,0)); lc+=(int)pp.contour_ids.size(); }
					 lc=0;
					 for(const auto& pp:fg2.l2_side){ solidpolygon P; for(int k=0;k<(int)pp.contour_ids.size();++k) P.contours.push_back(lc+k); gp_l2g.push_back(P); gr_l2g.push_back(polygon_reference((int)gp_l2g.size()-1,1)); lc+=(int)pp.contour_ids.size(); }
					}
					SolidRegionGrid gg_l1g,gg_l2g;
					gg_l1g.precompute(gr_l1g,gp_l1g,gc_l1g);
					gg_l2g.precompute(gr_l2g,gp_l2g,gc_l2g);

					std::vector<t1_t2_tet_info> gt1g,gt2g;
					std::vector<t12_tet_info>   gt12g;

					for(index_t tt=0;tt<nb_tp;++tt){
						const signed_index_t* v=del3p->cell_to_v()+4*tt;
						int cnt1=0;
						index_t vl1l[2]={},vl2l[2]={}; int cl1=0,cl2=0;
						for(int lv=0;lv<4;++lv){
							if(static_cast<index_t>(v[lv])<n_l1_pts){cnt1++;if(cl1<2)vl1l[cl1++]=static_cast<index_t>(v[lv]);}
							else{if(cl2<2)vl2l[cl2++]=static_cast<index_t>(v[lv]);}
						}
						if(cnt1==3||cnt1==1){
							bool in_l1=(cnt1==3);
							index_t bv[3]; int bc=0; index_t av=0;
							for(int lv=0;lv<4;++lv){
								bool ib=in_l1?(static_cast<index_t>(v[lv])<n_l1_pts):(static_cast<index_t>(v[lv])>=n_l1_pts);
								if(ib) bv[bc++]=static_cast<index_t>(v[lv]); else av=static_cast<index_t>(v[lv]);
							}
							double cx=(coords_3d[3*bv[0]]+coords_3d[3*bv[1]]+coords_3d[3*bv[2]])/3.0;
							double cy=(coords_3d[3*bv[0]+1]+coords_3d[3*bv[1]+1]+coords_3d[3*bv[2]+1])/3.0;
							const SolidRegionGrid& bg2=in_l1?gg_l1g:gg_l2g;
							if(bg2.classify(cx,cy)<0) continue;

							t1_t2_tet_info info; info.type=in_l1?1:2;
							info.base_verts[0]=bv[0];info.base_verts[1]=bv[1];info.base_verts[2]=bv[2]; info.apex_vert=av;
							for(int lv=0;lv<4;++lv) info.orig_verts[lv]=static_cast<index_t>(v[lv]);
							for(int lf=0;lf<4;++lf) info.faces[lf]={to_global(v[fvp2[lf][0]]),to_global(v[fvp2[lf][1]]),to_global(v[fvp2[lf][2]])};
							info.centroid_x=cx; info.centroid_y=cy; info.centroid_in_solid=true;
							info.fails_midpoint=false; info.void_l1_count=0; info.void_l2_count=0;
							{
								const double st[3]={2./5.,0.5,3./5.}; bool av1=true,av2=true;
								for(int be=0;be<3;++be){
									double bx=coords_3d[3*info.base_verts[be]],by=coords_3d[3*info.base_verts[be]+1];
									double ax=coords_3d[3*info.apex_vert],ay=coords_3d[3*info.apex_vert+1];
									for(int si=0;si<3;++si){
										double mx=bx+st[si]*(ax-bx),my=by+st[si]*(ay-by);
										info.midpoint_x[be][si]=mx; info.midpoint_y[be][si]=my;
										info.midpoint_in_solid_l1[be][si]=(gg_l1g.classify(mx,my)>=0);
										info.midpoint_in_solid_l2[be][si]=(gg_l2g.classify(mx,my)>=0);
										if(!info.midpoint_in_solid_l1[be][si]) info.void_l1_count++;
										if(!info.midpoint_in_solid_l2[be][si]) info.void_l2_count++;
										if(info.midpoint_in_solid_l1[be][si]) av1=false;
										if(info.midpoint_in_solid_l2[be][si]) av2=false;
									}
								}
								if(av1&&av2) info.fails_midpoint=true;
							}
							if(in_l1) gt1g.push_back(info); else gt2g.push_back(info);
						}
						else if(cnt1==2){
							double mx1=(coords_3d[3*vl1l[0]]+coords_3d[3*vl1l[1]])*0.5;
							double my1=(coords_3d[3*vl1l[0]+1]+coords_3d[3*vl1l[1]+1])*0.5;
							double mx2=(coords_3d[3*vl2l[0]]+coords_3d[3*vl2l[1]])*0.5;
							double my2=(coords_3d[3*vl2l[0]+1]+coords_3d[3*vl2l[1]+1])*0.5;
							int r1=gg_l1g.classify(mx1,my1);
							if(r1<0){Eigen::Vector2d m1v(mx1,my1);if(point_near_contour_boundary(m1v,gr_l1g,gp_l1g,gc_l1g)) r1=0;}
							int r2=gg_l2g.classify(mx2,my2);
							if(r2<0){Eigen::Vector2d m2v(mx2,my2);if(point_near_contour_boundary(m2v,gr_l2g,gp_l2g,gc_l2g)) r2=0;}
							if(r1<0||r2<0) continue;
							t12_tet_info info; info.fails_midpoint=false; info.fails_isolated=false; info.fails_group_affinity=false;
							for(int lv=0;lv<4;++lv) info.verts[lv]=to_global(static_cast<index_t>(v[lv]));
							for(int lf=0;lf<4;++lf) info.faces[lf]={to_global(v[fvp2[lf][0]]),to_global(v[fvp2[lf][1]]),to_global(v[fvp2[lf][2]])};
							gt12g.push_back(info);
						}
					}

					std::vector<std::array<index_t,3>> gfv_t1g,gfv_t2g;
					for(const auto& info:gt1g){
						bool valid=info.centroid_in_solid&&!(global_filter_t1_midpoint&&info.fails_midpoint);
						if(valid){ cnt_t1v++;
							tv_t1.push_back({to_global(info.orig_verts[0]),to_global(info.orig_verts[1]),to_global(info.orig_verts[2]),to_global(info.orig_verts[3])});
							for(int lf=0;lf<4;++lf){ fv_t1.push_back(info.faces[lf]); gfv_t1g.push_back(info.faces[lf]); }
						}
					}
					for(const auto& info:gt2g){
						bool valid=info.centroid_in_solid&&!(global_filter_t2_midpoint&&info.fails_midpoint);
						if(valid){ cnt_t2v++;
							tv_t2.push_back({to_global(info.orig_verts[0]),to_global(info.orig_verts[1]),to_global(info.orig_verts[2]),to_global(info.orig_verts[3])});
							for(int lf=0;lf<4;++lf){ fv_t2.push_back(info.faces[lf]); gfv_t2g.push_back(info.faces[lf]); }
						}
					}
					// isolated filter for this group's T12
					{
						std::unordered_set<std::array<index_t,3>,FaceHashP2> rf2;
						for(const auto& f:gfv_t1g) rf2.insert(make_fk_p2(f[0],f[1],f[2]));
						for(const auto& f:gfv_t2g) rf2.insert(make_fk_p2(f[0],f[1],f[2]));
						size_t n12g=gt12g.size();
						std::vector<bool> reach(n12g,false);
						bool fc=true;
						while(fc){ fc=false;
							for(size_t i=0;i<n12g;++i){
								if(gt12g[i].fails_midpoint||reach[i]) continue;
								for(int lf=0;lf<4;++lf)
									if(rf2.count(make_fk_p2(gt12g[i].faces[lf][0],gt12g[i].faces[lf][1],gt12g[i].faces[lf][2])))
										{reach[i]=true;for(int lf2=0;lf2<4;++lf2) rf2.insert(make_fk_p2(gt12g[i].faces[lf2][0],gt12g[i].faces[lf2][1],gt12g[i].faces[lf2][2]));fc=true;break;}
							}
						}
						for(size_t i=0;i<n12g;++i) if(!reach[i]) gt12g[i].fails_isolated=true;
					}
					for(const auto& info:gt12g){
						if(!info.fails_midpoint&&!info.fails_isolated&&!info.fails_group_affinity){
							cnt_t12v++;
							tv_t12.push_back({info.verts[0],info.verts[1],info.verts[2],info.verts[3]});
							for(int lf=0;lf<4;++lf) fv_t12.push_back(info.faces[lf]);
						}
					}
					std::cout << "      T1v=" << cnt_t1v << " T2v=" << cnt_t2v << " T12v=" << cnt_t12v << "\n";
				} // end group loop
			}
			// ============================================================
			// ALGO 1 / 2 / 4: single Delaunay per pair, full T1/T2/T12 filters
			// ============================================================
			else {
				Delaunay_var delaunay3d = Delaunay::create(3, "BDEL");
				delaunay3d->set_vertices(N3d, coords_3d.data());
				index_t nb_tets = delaunay3d->nb_cells();
				std::cout << "  Tets=" << nb_tets << "\n";

				std::vector<t1_t2_tet_info> loc_t1, loc_t2;
				std::vector<t12_tet_info> t12_pair;

				for (index_t tt = 0; tt < nb_tets; ++tt) {
					const signed_index_t* v = delaunay3d->cell_to_v() + 4 * tt;
					int cnt1 = 0;
					index_t vl1[2]={}, vl2[2]={}; int cl1=0, cl2=0;
					for (int lv=0;lv<4;++lv) {
						if(static_cast<index_t>(v[lv])<n_l1_pts){cnt1++;if(cl1<2)vl1[cl1++]=static_cast<index_t>(v[lv]);}
						else{if(cl2<2)vl2[cl2++]=static_cast<index_t>(v[lv]);}
					}
					if (cnt1==3||cnt1==1) {
						bool in_l1=(cnt1==3);
						t1_t2_tet_info info; info.type=in_l1?1:2;
						int bc=0; info.apex_vert=0;
						for(int lv=0;lv<4;++lv){
							info.orig_verts[lv]=static_cast<index_t>(v[lv]);
							bool ib=in_l1?(static_cast<index_t>(v[lv])<n_l1_pts):(static_cast<index_t>(v[lv])>=n_l1_pts);
							if(ib) info.base_verts[bc++]=static_cast<index_t>(v[lv]); else info.apex_vert=static_cast<index_t>(v[lv]);
						}
						for(int lf=0;lf<4;++lf)
							info.faces[lf]={to_global(v[fvp2[lf][0]]),to_global(v[fvp2[lf][1]]),to_global(v[fvp2[lf][2]])};
						info.centroid_x=(coords_3d[3*info.base_verts[0]]+coords_3d[3*info.base_verts[1]]+coords_3d[3*info.base_verts[2]])/3.0;
						info.centroid_y=(coords_3d[3*info.base_verts[0]+1]+coords_3d[3*info.base_verts[1]+1]+coords_3d[3*info.base_verts[2]+1])/3.0;
						const SolidRegionGrid& bg=in_l1?pc.grid_l1:pc.grid_l2;
						info.centroid_in_solid=(bg.classify(info.centroid_x,info.centroid_y)>=0);
						info.fails_midpoint=false; info.void_l1_count=0; info.void_l2_count=0;
						{
							const double st[3]={2./5.,0.5,3./5.}; bool av1=true,av2=true;
							for(int be=0;be<3;++be){
								double bx=coords_3d[3*info.base_verts[be]],by=coords_3d[3*info.base_verts[be]+1];
								double ax=coords_3d[3*info.apex_vert],ay=coords_3d[3*info.apex_vert+1];
								for(int si=0;si<3;++si){
									double mx=bx+st[si]*(ax-bx),my=by+st[si]*(ay-by);
									info.midpoint_x[be][si]=mx; info.midpoint_y[be][si]=my;
									info.midpoint_in_solid_l1[be][si]=(pc.grid_l1.classify(mx,my)>=0);
									info.midpoint_in_solid_l2[be][si]=(pc.grid_l2.classify(mx,my)>=0);
									if(!info.midpoint_in_solid_l1[be][si]) info.void_l1_count++;
									if(!info.midpoint_in_solid_l2[be][si]) info.void_l2_count++;
									if(info.midpoint_in_solid_l1[be][si]) av1=false;
									if(info.midpoint_in_solid_l2[be][si]) av2=false;
								}
							}
							if(av1&&av2) info.fails_midpoint=true;
						}
						if(in_l1) loc_t1.push_back(info); else loc_t2.push_back(info);
					}
					else if (cnt1==2) {
						t12_tet_info info; info.fails_midpoint=false; info.fails_isolated=false; info.fails_group_affinity=false;
						for(int lv=0;lv<4;++lv) info.verts[lv]=to_global(static_cast<index_t>(v[lv]));
						for(int lf=0;lf<4;++lf)
							info.faces[lf]={to_global(v[fvp2[lf][0]]),to_global(v[fvp2[lf][1]]),to_global(v[fvp2[lf][2]])};
						{
							double mx1=(coords_3d[3*vl1[0]]+coords_3d[3*vl1[1]])*0.5;
							double my1=(coords_3d[3*vl1[0]+1]+coords_3d[3*vl1[1]+1])*0.5;
							Eigen::Vector2d m1v(mx1,my1);
							int r1=pc.grid_l1.classify(mx1,my1);
							if(r1<0&&point_near_contour_boundary(m1v,pc.prefs_l1,pc.polys,levels[idxL1].contours)) r1=0;
							if(r1<0){info.fails_midpoint=true;}
							else{
								double mx2=(coords_3d[3*vl2[0]]+coords_3d[3*vl2[1]])*0.5;
								double my2=(coords_3d[3*vl2[0]+1]+coords_3d[3*vl2[1]+1])*0.5;
								Eigen::Vector2d m2v(mx2,my2);
								int r2=pc.grid_l2.classify(mx2,my2);
								if(r2<0&&point_near_contour_boundary(m2v,pc.prefs_l2,pc.polys,levels[idxL2].contours)) r2=0;
								if(r2<0) info.fails_midpoint=true;
							}
						}
						t12_pair.push_back(info);
					}
				}

				// T1/T2 valid faces
				std::vector<std::array<index_t,3>> fv_t1_loc, fv_t2_loc;
				for (const auto& info:loc_t1){
					bool valid=info.centroid_in_solid&&!(global_filter_t1_midpoint&&info.fails_midpoint);
					if(valid){ cnt_t1v++;
						tv_t1.push_back({to_global(info.orig_verts[0]),to_global(info.orig_verts[1]),to_global(info.orig_verts[2]),to_global(info.orig_verts[3])});
						for(int lf=0;lf<4;++lf){fv_t1.push_back(info.faces[lf]);fv_t1_loc.push_back(info.faces[lf]);}
					}
				}
				for (const auto& info:loc_t2){
					bool valid=info.centroid_in_solid&&!(global_filter_t2_midpoint&&info.fails_midpoint);
					if(valid){ cnt_t2v++;
						tv_t2.push_back({to_global(info.orig_verts[0]),to_global(info.orig_verts[1]),to_global(info.orig_verts[2]),to_global(info.orig_verts[3])});
						for(int lf=0;lf<4;++lf){fv_t2.push_back(info.faces[lf]);fv_t2_loc.push_back(info.faces[lf]);}
					}
				}

				// Isolated filter + IMA rescue for T12
				{
					std::unordered_set<std::array<index_t,3>,FaceHashP2> rf3;
					for(const auto& f:fv_t1_loc) rf3.insert(make_fk_p2(f[0],f[1],f[2]));
					for(const auto& f:fv_t2_loc) rf3.insert(make_fk_p2(f[0],f[1],f[2]));
					size_t n12=t12_pair.size();
					std::vector<bool> reach(n12,false);
					bool fc=true;
					while(fc){ fc=false;
						for(size_t i=0;i<n12;++i){
							if(t12_pair[i].fails_midpoint||reach[i]) continue;
							for(int lf=0;lf<4;++lf)
								if(rf3.count(make_fk_p2(t12_pair[i].faces[lf][0],t12_pair[i].faces[lf][1],t12_pair[i].faces[lf][2])))
									{reach[i]=true;for(int lf2=0;lf2<4;++lf2) rf3.insert(make_fk_p2(t12_pair[i].faces[lf2][0],t12_pair[i].faces[lf2][1],t12_pair[i].faces[lf2][2]));fc=true;break;}
						}
					}
					for(size_t i=0;i<n12;++i) if(!reach[i]) t12_pair[i].fails_isolated=true;

					// IMA rescue using local coords (no V3d_multi dependency)
					for(size_t i=0;i<n12;++i){
						if(!t12_pair[i].fails_isolated||t12_pair[i].fails_midpoint) continue;
						index_t vl1r[2]={},vl2r[2]={}; int c1r=0,c2r=0;
						for(int lv=0;lv<4;++lv){
							index_t gv=t12_pair[i].verts[lv];
							if(gv>=lv_offset[idxL1]&&gv<lv_offset[idxL1]+n_l1_pts){if(c1r<2)vl1r[c1r++]=gv;}
							else{if(c2r<2)vl2r[c2r++]=gv;}
						}
						if(c1r!=2||c2r!=2) continue;
						bool l1ima=false; for(int k=0;k<2;++k) if(vl1r[k]>=g_ima_s_l1&&vl1r[k]<g_ima_e_l1) l1ima=true;
						bool l2ima=false; for(int k=0;k<2;++k) if(vl2r[k]>=g_ima_s_l2&&vl2r[k]<g_ima_e_l2) l2ima=true;
						if(!l1ima&&!l2ima) continue;
						bool rescue=false;
						if(l1ima){
							index_t loc0=vl1r[0]-lv_offset[idxL1], loc1=vl1r[1]-lv_offset[idxL1];
							double mx=(coords_3d[3*loc0]+coords_3d[3*loc1])*0.5;
							double my=(coords_3d[3*loc0+1]+coords_3d[3*loc1+1])*0.5;
							if(pc.grid_l1.classify(mx,my)>=0&&pc.grid_l2.classify(mx,my)>=0) rescue=true;
						}
						if(!rescue&&l2ima){
							index_t c0=n_l1_pts+(vl2r[0]-lv_offset[idxL2]), c1n=n_l1_pts+(vl2r[1]-lv_offset[idxL2]);
							double mx=(coords_3d[3*c0]+coords_3d[3*c1n])*0.5;
							double my=(coords_3d[3*c0+1]+coords_3d[3*c1n+1])*0.5;
							if(pc.grid_l1.classify(mx,my)>=0&&pc.grid_l2.classify(mx,my)>=0) rescue=true;
						}
						if(rescue) t12_pair[i].fails_isolated=false;
					}
				}

				for(const auto& info:t12_pair){
					if(!info.fails_midpoint&&!info.fails_isolated&&!info.fails_group_affinity){
						cnt_t12v++;
						tv_t12.push_back({info.verts[0],info.verts[1],info.verts[2],info.verts[3]});
						for(int lf=0;lf<4;++lf) fv_t12.push_back(info.faces[lf]);
					}
				}
			} // end algo 1/2/4 branch

			std::cout << "  T1v=" << cnt_t1v << " T2v=" << cnt_t2v << " T12v=" << cnt_t12v << "\n";

			accum_faces_t1_valid.insert(accum_faces_t1_valid.end(), fv_t1.begin(), fv_t1.end());
			accum_faces_t2_valid.insert(accum_faces_t2_valid.end(), fv_t2.begin(), fv_t2.end());
			accum_faces_t12_valid.insert(accum_faces_t12_valid.end(), fv_t12.begin(), fv_t12.end());
			accum_tet_verts.insert(accum_tet_verts.end(), tv_t1.begin(), tv_t1.end());
			accum_tet_verts.insert(accum_tet_verts.end(), tv_t2.begin(), tv_t2.end());
			accum_tet_verts.insert(accum_tet_verts.end(), tv_t12.begin(), tv_t12.end());
		}

		// All valid faces
		std::vector<std::array<index_t, 3>> all_valid_faces;
		all_valid_faces.insert(all_valid_faces.end(), accum_faces_t1_valid.begin(), accum_faces_t1_valid.end());
		all_valid_faces.insert(all_valid_faces.end(), accum_faces_t2_valid.begin(), accum_faces_t2_valid.end());
		all_valid_faces.insert(all_valid_faces.end(), accum_faces_t12_valid.begin(), accum_faces_t12_valid.end());

		std::cout << "\n=================================\n";
		std::cout << "MULTI-SECTION TOTALS\n";
		std::cout << "  Total vertices: " << total_global_pts << "\n";
		std::cout << "  T1 valid faces: " << accum_faces_t1_valid.size() << "\n";
		std::cout << "  T2 valid faces: " << accum_faces_t2_valid.size() << "\n";
		std::cout << "  T12 valid faces: " << accum_faces_t12_valid.size() << "\n";
		std::cout << "  All valid faces: " << all_valid_faces.size() << "\n";

		// Free boundary using igl::boundary_facets
		// Build T_all directly from stored tet vertices with proper orientation
		{
			int n_all_tets = static_cast<int>(accum_tet_verts.size());
			Eigen::MatrixXi T_all(n_all_tets, 4);
			for (int ti = 0; ti < n_all_tets; ++ti) {
				for (int lv = 0; lv < 4; ++lv)
					T_all(ti, lv) = static_cast<int>(accum_tet_verts[ti][lv]);
			}

			// Orient all tets positively: if det < 0, swap verts 0 and 1
			int reoriented = 0;
			for (int r = 0; r < n_all_tets; ++r) {
				int i0 = T_all(r, 0), i1 = T_all(r, 1), i2 = T_all(r, 2), i3 = T_all(r, 3);
				Eigen::Vector3d p0 = V3d_multi.row(i0);
				Eigen::Vector3d p1 = V3d_multi.row(i1);
				Eigen::Vector3d p2 = V3d_multi.row(i2);
				Eigen::Vector3d p3 = V3d_multi.row(i3);
				double det = (p1 - p0).dot((p2 - p0).cross(p3 - p0));
				if (det < 0) {
					std::swap(T_all(r, 0), T_all(r, 1));
					reoriented++;
				}
			}
			std::cout << "  Tets for boundary_facets: " << n_all_tets << "\n";
			std::cout << "  Reoriented tets (negative det): " << reoriented << " / " << n_all_tets << "\n";

			Eigen::MatrixXi F_fb;
			igl::boundary_facets(T_all, F_fb);

			// Remove horizontal caps at INTERIOR shared levels.
			// The 3D Delaunay of pair (Li-1,Li) and pair (Li,Li+1) may produce
			// different 2D triangulations of aug[Li] even when aug[Li] is identical,
			// because the L1/L3 points influence the Delaunay differently.
			// These uncanceled caps appear as internal "floors" in the skin.
			// Fix: after boundary_facets, drop every face whose 3 vertices all share
			// the same z-value that belongs to an INTERIOR level of the range.
			{
				// Build set of interior z-values (not the first nor last level)
				std::set<double> interior_z;
				for (int li = rangeStart; li < rangeEnd - 1; ++li)
					interior_z.insert(levels[li].zCoord);

				if (!interior_z.empty()) {
					std::vector<int> keep;
					keep.reserve(F_fb.rows());
					for (int i = 0; i < F_fb.rows(); ++i) {
						double z0 = V3d_multi(F_fb(i, 0), 2);
						double z1 = V3d_multi(F_fb(i, 1), 2);
						double z2 = V3d_multi(F_fb(i, 2), 2);
						// Face is a horizontal interior cap only if all 3 z-values match
						// the same interior level (within floating-point tolerance)
						bool is_interior_cap = false;
						for (double iz : interior_z) {
							if (std::abs(z0 - iz) < 1e-9 &&
								std::abs(z1 - iz) < 1e-9 &&
								std::abs(z2 - iz) < 1e-9) {
								is_interior_cap = true;
								break;
							}
						}
						if (!is_interior_cap)
							keep.push_back(i);
					}
					int removed_caps = F_fb.rows() - static_cast<int>(keep.size());
					if (removed_caps > 0) {
						Eigen::MatrixXi F_clean(static_cast<int>(keep.size()), 3);
						for (int i = 0; i < static_cast<int>(keep.size()); ++i)
							F_clean.row(i) = F_fb.row(keep[i]);
						F_fb = F_clean;
						std::cout << "  Removed " << removed_caps
							<< " interior horizontal cap faces at shared levels\n";
					}
				}
			}

			global_faces_free_boundary.clear();
			for (int i = 0; i < F_fb.rows(); ++i)
				global_faces_free_boundary.push_back({static_cast<index_t>(F_fb(i,0)), static_cast<index_t>(F_fb(i,1)), static_cast<index_t>(F_fb(i,2))});
			delaunay3d_free_boundary_faces = static_cast<int>(F_fb.rows());
			std::cout << "  Free boundary faces (after cap removal): " << delaunay3d_free_boundary_faces << "\n";

			// Reindex vertices for manifold check (no unreferenced vertices)
			std::map<int, int> old_to_new;
			int next_id = 0;
			Eigen::MatrixXi F_fb_reindexed(delaunay3d_free_boundary_faces, 3);
			for (int i = 0; i < delaunay3d_free_boundary_faces; ++i) {
				for (int lv = 0; lv < 3; ++lv) {
					int old_v = F_fb(i, lv);
					if (old_to_new.find(old_v) == old_to_new.end())
						old_to_new[old_v] = next_id++;
					F_fb_reindexed(i, lv) = old_to_new[old_v];
				}
			}

			global_manifold_info = check_manifold(F_fb_reindexed);

			std::cout << "\n  === MANIFOLD DIAGNOSTIC ===\n";
			std::cout << "  Edge-manifold:   " << (global_manifold_info.edge_manifold ? "YES" : "NO") << "\n";
			std::cout << "  Vertex-manifold: " << (global_manifold_info.vertex_manifold ? "YES" : "NO") << "\n";
			std::cout << "  Euler characteristic: " << global_manifold_info.euler_characteristic << "\n";
			if (!global_manifold_info.vertex_manifold)
				std::cout << "  Non-manifold vertices: " << global_manifold_info.non_manifold_vertices << "\n";
			if (!global_manifold_info.edge_manifold)
				std::cout << "  Non-manifold edges: " << global_manifold_info.non_manifold_edges << "\n";
			std::cout << "  Boundary vertices: " << next_id << "\n";
			std::cout << "  Result: " << (global_manifold_info.is_manifold() ? "MANIFOLD" : "NOT MANIFOLD") << "\n";
			std::cout << "  =========================\n";
		}
		std::cout << "=================================\n";

		// Store for viewer setup later
		global_V3d_multi = V3d_multi;
		global_accum_faces_t1_valid = accum_faces_t1_valid;
		global_accum_faces_t2_valid = accum_faces_t2_valid;
		global_accum_faces_t12_valid = accum_faces_t12_valid;
		global_all_valid_faces = all_valid_faces;
		global_multi_range_start = rangeStart;
		global_multi_range_end = rangeEnd;
	}

	// =========================
	// VIEWER l1
	// =========================
	igl::opengl::glfw::Viewer viewer_l1;
	igl::opengl::glfw::imgui::ImGuiPlugin plugin_l1;
	igl::opengl::glfw::imgui::ImGuiMenu menu_l1;

	viewer_l1.plugins.push_back(&plugin_l1);
	plugin_l1.widgets.push_back(&menu_l1);

	viewer_l1.core().background_color << 1.0, 1.0, 1.0, 1.0;
	viewer_l1.data().clear();

	// =========================
	// VIEWER l2
	// =========================
	igl::opengl::glfw::Viewer viewer_l2;
	igl::opengl::glfw::imgui::ImGuiPlugin plugin_l2;
	igl::opengl::glfw::imgui::ImGuiMenu menu_l2;

	viewer_l2.plugins.push_back(&plugin_l2);
	plugin_l2.widgets.push_back(&menu_l2);

	viewer_l2.core().background_color << 1.0, 1.0, 1.0, 1.0;
	viewer_l2.data().clear();

	labels.clear();
	labels_l2.clear();

	if (choice == 'N' || choice == 'n') {
	// Compute bounding box for auto-scaling axis length and label offsets
	double auto_bb_edge = 1.0, auto_axis_len = 1.0, auto_label_off = 0.03;
	{
		double _mix = 1e30, _miy = 1e30, _max = -1e30, _may = -1e30;
		for (const auto& _c : levels[startIdx - 1].contours)
			for (const auto& _p : _c.points) {
				_mix = std::min(_mix, _p.x()); _miy = std::min(_miy, _p.y());
				_max = std::max(_max, _p.x()); _may = std::max(_may, _p.y());
			}
		for (const auto& _c : levels[startIdx].contours)
			for (const auto& _p : _c.points) {
				_mix = std::min(_mix, _p.x()); _miy = std::min(_miy, _p.y());
				_max = std::max(_max, _p.x()); _may = std::max(_may, _p.y());
			}
		auto_bb_edge   = std::max({_max - _mix, _may - _miy, 1.0});
		auto_axis_len  = 0.20 * auto_bb_edge;
		auto_label_off = 0.03 * auto_bb_edge;
	}
	// =======================
	// DIBUJO NIVEL l1
	// =======================

	int cL1 = 0;
	const auto& section_l1 = levels[startIdx - 1];

	for (const auto& contour : section_l1.contours) {

		int numVertices = static_cast<int>(contour.points.size());
		if (numVertices == 0) continue;

		Eigen::MatrixXd V(numVertices, 3);

		for (int j = 0; j < numVertices; ++j) {
			V.row(j) = contour.points[j];
		}

		Eigen::Vector3f posLabel(
			static_cast<float>(contour.points[0].x() - auto_label_off),
			static_cast<float>(contour.points[0].y()),
			static_cast<float>(contour.points[0].z())
		);

		char letter = 'A' + cL1++;
		labels.push_back({ posLabel, std::string(1, letter) });

		viewer_l1.data().add_points(V, Eigen::RowVector3d(1, 0, 0));

		for (int j = 0; j < numVertices; ++j) {
			viewer_l1.data().add_edges(
				V.row(j),
				V.row((j + 1) % numVertices),
				Eigen::RowVector3d(0, 0, 0)
			);
		}
	}

	// =======================
	// DIBUJO NIVEL l2
	// =======================

	int cL2 = 0;
	const auto& section_l2 = levels[startIdx];

	for (const auto& contour : section_l2.contours) {

		int numVertices = static_cast<int>(contour.points.size());
		if (numVertices == 0) continue;

		Eigen::MatrixXd V(numVertices, 3);

		for (int j = 0; j < numVertices; ++j) {
			V.row(j) = contour.points[j];
		}

		Eigen::Vector3f posLabel(
			static_cast<float>(contour.points[0].x() - auto_label_off),
			static_cast<float>(contour.points[0].y()),
			static_cast<float>(contour.points[0].z())
		);

		labels_l2.push_back({ posLabel, std::to_string(++cL2) });

		viewer_l2.data().add_points(V, Eigen::RowVector3d(1, 0, 0));

		for (int j = 0; j < numVertices; ++j) {
			viewer_l2.data().add_edges(
				V.row(j),
				V.row((j + 1) % numVertices),
				Eigen::RowVector3d(0, 0, 0)
			);
		}
	}

	if (!levels.empty()) {

		double minX = std::numeric_limits<double>::infinity();
		double minY = std::numeric_limits<double>::infinity();
		double pivotZ = levels[startIdx - 1].zCoord;

		for (int i = startIdx - 1; i < endIdx; ++i) {

			for (const auto& contour : levels[i].contours) {

				for (const auto& point : contour.points) {

					if (point.x() < minX) minX = point.x();
					if (point.y() < minY) minY = point.y();
				}
			}
		}

		Eigen::RowVector3d pivot(minX, minY, pivotZ);

		double axisLength = auto_axis_len;

		Eigen::RowVector3d xAxisEnd = pivot + Eigen::RowVector3d(axisLength, 0, 0);
		Eigen::RowVector3d yAxisEnd = pivot + Eigen::RowVector3d(0, axisLength, 0);
		Eigen::RowVector3d zAxisEnd = pivot + Eigen::RowVector3d(0, 0, axisLength);

		viewer_l1.data().add_edges(pivot, xAxisEnd, Eigen::RowVector3d(0, 0, 0));
		viewer_l1.data().add_edges(pivot, yAxisEnd, Eigen::RowVector3d(0.5, 0, 0.5));
		viewer_l1.data().add_edges(pivot, zAxisEnd, Eigen::RowVector3d(0, 0, 1));
		viewer_l1.data().add_points(pivot, Eigen::RowVector3d(0, 0, 0));

		viewer_l2.data().add_edges(pivot, xAxisEnd, Eigen::RowVector3d(0, 0, 0));
		viewer_l2.data().add_edges(pivot, yAxisEnd, Eigen::RowVector3d(0.5, 0, 0.5));
		viewer_l2.data().add_edges(pivot, zAxisEnd, Eigen::RowVector3d(0, 0, 1));
		viewer_l2.data().add_points(pivot, Eigen::RowVector3d(0, 0, 0));
	}
	} // end if (choice == 'N') for single-pair drawing

	menu_l1.callback_draw_viewer_window = [&]() {

		ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(1.0f, 1.0f, 1.0f, 1.0f));
		ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.0f, 0.0f, 0.0f, 1.0f));
		ImGui::PushStyleColor(ImGuiCol_TitleBg, ImVec4(0.85f, 0.85f, 0.85f, 1.0f));
		ImGui::PushStyleColor(ImGuiCol_TitleBgActive, ImVec4(0.75f, 0.75f, 0.75f, 1.0f));

		ImGui::Begin("L1");
		ImGui::SetWindowFontScale(1.6f);

		ImGui::Text("Contours:");
		ImGui::SameLine();

		for (size_t i = 0; i < forest_l1.size(); ++i) {

			char letter = 'A' + forest_l1[i].contourId;

			ImGui::Text("%c%s",
				letter,
				(i + 1 < forest_l1.size() ? "," : "")
			);
			ImGui::SameLine();
		}
		ImGui::NewLine();

		ImGui::Text("Polygons:");
		ImGui::SameLine();

		for (size_t p = 0; p < polygon_refs_l1.size(); ++p) {
			const auto& poly = all_polygons[polygon_refs_l1[p].polygon_id];

			ImGui::Text("(");
			ImGui::SameLine();

			for (size_t j = 0; j < poly.contours.size(); ++j) {

				int cid = poly.contours[j];
				char letter = 'A' + forest_l1[cid].contourId;

				ImGui::Text("%c%s",
					letter,
					(j + 1 < poly.contours.size() ? "," : "")
				);
				ImGui::SameLine();
			}

			ImGui::Text(")");

			if (p + 1 < polygon_refs_l1.size()) {
				ImGui::SameLine();
				ImGui::Text(",");
				ImGui::SameLine();
			}
		}
		ImGui::NewLine();

		// Ã¢Å“â€¦ NUEVO: Mostrar estructura del bosque
		ImGui::Text("Forest:");
		ImGui::SameLine();
		ImGui::TextWrapped("%s", global_forest_text_l1.c_str());

		ImGui::SetWindowFontScale(1.0f);
		ImGui::End();

		ImGui::PopStyleColor(4);

		drawForestDiagramForest(forest_l1, true);

		ImGui::SetWindowFontScale(1.8f);
		for (const auto& lbl : labels) {

			auto screen = projectToScreen(lbl.position, viewer_l1);

			ImGui::GetForegroundDrawList()->AddText(
				ImVec2(screen.x(), screen.y()),
				IM_COL32(0, 0, 0, 255),
				lbl.text.c_str()
			);
		}
		ImGui::SetWindowFontScale(1.0f);
	};

	menu_l2.callback_draw_viewer_window = [&]() {

		ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(1.0f, 1.0f, 1.0f, 1.0f));
		ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.0f, 0.0f, 0.0f, 1.0f));
		ImGui::PushStyleColor(ImGuiCol_TitleBg, ImVec4(0.85f, 0.85f, 0.85f, 1.0f));
		ImGui::PushStyleColor(ImGuiCol_TitleBgActive, ImVec4(0.75f, 0.75f, 0.75f, 1.0f));

		ImGui::Begin("L2");
		ImGui::SetWindowFontScale(1.6f);

		ImGui::Text("Contours:");
		ImGui::SameLine();

		for (size_t i = 0; i < forest_l2.size(); ++i) {

		 int number = forest_l2[i].contourId + 1;

			ImGui::Text("%d%s",
				number,
				(i + 1 < forest_l2.size() ? "," : "")
			);
			ImGui::SameLine();
		}
		ImGui::NewLine();

		ImGui::Text("Polygons:");
		ImGui::SameLine();

		for (size_t p = 0; p < polygon_refs_l2.size(); ++p) {
			const auto& poly = all_polygons[polygon_refs_l2[p].polygon_id];

			ImGui::Text("(");
			ImGui::SameLine();

			for (size_t j = 0; j < poly.contours.size(); ++j) {

			 int cid = poly.contours[j];
			 int number = forest_l2[cid].contourId + 1;

				ImGui::Text("%d%s",
					number,
					(j + 1 < poly.contours.size() ? "," : "")
				);
				ImGui::SameLine();
			}

			ImGui::Text(")");

			if (p + 1 < polygon_refs_l2.size()) {
				ImGui::SameLine();
				ImGui::Text(",");
				ImGui::SameLine();
			}
		}
		ImGui::NewLine();

		// Ã¢Å“â€¦ NUEVO: Mostrar estructura del bosque
		ImGui::Text("Forest:");
		ImGui::SameLine();
		ImGui::TextWrapped("%s", global_forest_text_l2.c_str());

		ImGui::SetWindowFontScale(1.0f);
		ImGui::End();

		ImGui::PopStyleColor(4);

		drawForestDiagramForest(forest_l2, false);

		ImGui::SetWindowFontScale(1.8f);
		for (const auto& lbl : labels_l2) {

			auto screen = projectToScreen(lbl.position, viewer_l2);

			ImGui::GetForegroundDrawList()->AddText(
				ImVec2(screen.x(), screen.y()),
				IM_COL32(0, 0, 0, 255),
				lbl.text.c_str()
			);
		}
		ImGui::SetWindowFontScale(1.0f);
	};

	// ===============================
	// BOUNDING BOX
	// ===============================
	double minX = 1e30, minY = 1e30;
	double maxX = -1e30, maxY = -1e30;

	for (int i = startIdx - 1; i < endIdx; ++i) {

		for (const auto& contour : levels[i].contours) {

			for (const auto& p : contour.points) {

				minX = std::min(minX, p.x());
				minY = std::min(minY, p.y());
				maxX = std::max(maxX, p.x());
				maxY = std::max(maxY, p.y());
			}
		}
	}

	Eigen::MatrixXd Vbox(2, 3);
	Eigen::MatrixXi Fbox(0, 3);

	Vbox.row(0) << minX, minY, 0;
	Vbox.row(1) << maxX, maxY, 0;

	viewer_l1.data().set_mesh(Vbox, Fbox);
	viewer_l1.data().show_faces = false;

	viewer_l2.data().set_mesh(Vbox, Fbox);
	viewer_l2.data().show_faces = false;

	viewer_l1.data().point_size = 1.0;
	viewer_l2.data().point_size = 1.0;


	// =========================
	// VIEWER 3 - INTERSECCIÃƒâ€œN
	// =========================
	igl::opengl::glfw::Viewer viewer_intersection;
	igl::opengl::glfw::imgui::ImGuiPlugin plugin_inter;
	igl::opengl::glfw::imgui::ImGuiMenu menu_inter;

	viewer_intersection.plugins.push_back(&plugin_inter);
	plugin_inter.widgets.push_back(&menu_inter);

	viewer_intersection.core().background_color << 1.0, 1.0, 1.0, 1.0;
	viewer_intersection.data().clear();
	viewer_intersection.data().line_width = 2.0f;

	// Ã¢Å“â€¦ NUEVO: VIEWER 4 - SUPERPOSICIÃƒâ€œN SIN INTERSECCIÃƒâ€œN
	igl::opengl::glfw::Viewer viewer_overlay;
	igl::opengl::glfw::imgui::ImGuiPlugin plugin_overlay;
	igl::opengl::glfw::imgui::ImGuiMenu menu_overlay;

	viewer_overlay.plugins.push_back(&plugin_overlay);
	plugin_overlay.widgets.push_back(&menu_overlay);

	viewer_overlay.core().background_color << 1.0, 1.0, 1.0, 1.0;
	viewer_overlay.data().clear();
	viewer_overlay.data().line_width = 2.0f;

	// =========================
	// VIEWER 5 - HOLE vs HOLE
	// =========================
	igl::opengl::glfw::Viewer viewer_hh;
	igl::opengl::glfw::imgui::ImGuiPlugin plugin_hh;
	igl::opengl::glfw::imgui::ImGuiMenu menu_hh;

	viewer_hh.plugins.push_back(&plugin_hh);
	plugin_hh.widgets.push_back(&menu_hh);

	viewer_hh.core().background_color << 1.0, 1.0, 1.0, 1.0;
	viewer_hh.data().clear();
	viewer_hh.data().line_width = 2.0f;

	// =========================
	// VIEWER 6 - HOLE vs SOLID
	// =========================
	igl::opengl::glfw::Viewer viewer_hs;
	igl::opengl::glfw::imgui::ImGuiPlugin plugin_hs;
	igl::opengl::glfw::imgui::ImGuiMenu menu_hs;

	viewer_hs.plugins.push_back(&plugin_hs);
	plugin_hs.widgets.push_back(&menu_hs);

	viewer_hs.core().background_color << 1.0, 1.0, 1.0, 1.0;
	viewer_hs.data().clear();
	viewer_hs.data().line_width = 2.0f;

	// Multi-section viewer setup (after viewer declarations)
	if (choice == 'Y' || choice == 'y') {
		// Clear any previously set data
		viewer_l1.data().clear();
		viewer_intersection.data().clear();

		// VIEWER 1: All contours
		{
			const Eigen::RowVector3d color_contour(0, 0, 0);
			for (int li = global_multi_range_start - 1; li < global_multi_range_end; ++li) {
				for (const auto& contour : levels[li].contours) {
					int n = static_cast<int>(contour.points.size());
					if (n == 0) continue;
					Eigen::MatrixXd V(n, 3);
					for (int j = 0; j < n; ++j) V.row(j) = contour.points[j];
					for (int j = 0; j < n; ++j)
						viewer_l1.data().add_edges(V.row(j), V.row((j+1)%n), color_contour);
				}
			}
			viewer_l1.data().set_mesh(Vbox, Fbox);
			viewer_l1.data().show_faces = false;
		}

		// VIEWER 2: Free boundary
		{
			if (!global_faces_free_boundary.empty()) {
				int nf = static_cast<int>(global_faces_free_boundary.size());
				Eigen::MatrixXi F(nf, 3);
				for (int i = 0; i < nf; ++i)
					F.row(i) << static_cast<int>(global_faces_free_boundary[i][0]),
						static_cast<int>(global_faces_free_boundary[i][1]),
						static_cast<int>(global_faces_free_boundary[i][2]);
				viewer_intersection.data().set_mesh(global_V3d_multi, F);
				viewer_intersection.data().show_lines = true;
				viewer_intersection.data().show_faces = true;
				Eigen::MatrixXd C(nf, 3);
				for (int i = 0; i < nf; ++i) C.row(i) << 0.7, 0.7, 0.7;
				viewer_intersection.data().set_colors(C);
			} else {
				viewer_intersection.data().set_mesh(Vbox, Fbox);
				viewer_intersection.data().show_faces = false;
			}
			viewer_intersection.core().lighting_factor = 0.0f;
		}

		// Export free boundary as OBJ with reindexed vertices (no unreferenced)
		{
			int nf = static_cast<int>(global_faces_free_boundary.size());
			std::map<int, int> old_to_new;
			int next_id = 0;
			Eigen::MatrixXi F_fb(nf, 3);
			for (int i = 0; i < nf; ++i) {
				for (int lv = 0; lv < 3; ++lv) {
					int old_v = static_cast<int>(global_faces_free_boundary[i][lv]);
					if (old_to_new.find(old_v) == old_to_new.end())
						old_to_new[old_v] = next_id++;
					F_fb(i, lv) = old_to_new[old_v];
				}
			}
			Eigen::MatrixXd V_fb(next_id, 3);
			for (const auto& kv : old_to_new)
				V_fb.row(kv.second) = global_V3d_multi.row(kv.first);

			// Orient faces outward: flip faces whose normal points toward the centroid
			{
				Eigen::Vector3d centroid = V_fb.colwise().mean();
				int flipped = 0;
				for (int i = 0; i < F_fb.rows(); ++i) {
					Eigen::Vector3d a = V_fb.row(F_fb(i, 0));
					Eigen::Vector3d b = V_fb.row(F_fb(i, 1));
					Eigen::Vector3d c = V_fb.row(F_fb(i, 2));
					Eigen::Vector3d n = (b - a).cross(c - a);
					Eigen::Vector3d face_center = (a + b + c) / 3.0;
					if (n.dot(face_center - centroid) < 0) {
						std::swap(F_fb(i, 0), F_fb(i, 1));
						flipped++;
					}
				}
				std::cout << "Faces oriented outward: flipped " << flipped << " / " << F_fb.rows() << "\n";
			}

			// Also update global_faces_free_boundary to match
			{
				std::map<int, int> new_to_old;
				for (const auto& kv : old_to_new)
					new_to_old[kv.second] = kv.first;
				for (int i = 0; i < F_fb.rows(); ++i) {
					global_faces_free_boundary[i] = {
						static_cast<index_t>(new_to_old[F_fb(i, 0)]),
						static_cast<index_t>(new_to_old[F_fb(i, 1)]),
						static_cast<index_t>(new_to_old[F_fb(i, 2)])
					};
				}
			}

			std::string obj_path = "C:/Users/DELL/Desktop/Data/free_boundary.obj";
			if (igl::writeOBJ(obj_path, V_fb, F_fb))
				std::cout << "OBJ exported: " << obj_path << " ("
					<< V_fb.rows() << " vertices, " << nf << " faces)\n";
			else
				std::cerr << "ERROR: Could not write OBJ to " << obj_path << "\n";
		}

		// Callbacks for multi-section viewers
		menu_l1.callback_draw_viewer_window = [&]() {
			ImGui::Begin("All Contours");
			ImGui::SetWindowFontScale(1.4f);
			ImGui::Text("Sections %d to %d", global_multi_range_start, global_multi_range_end);
			ImGui::Text("Total levels: %d", global_multi_range_end - global_multi_range_start + 1);
			ImGui::End();
		};

		menu_inter.callback_draw_viewer_window = [&]() {
			ImGui::Begin("Free Boundary (Skin)");
			ImGui::SetWindowFontScale(1.4f);
			ImGui::Text("Surface of all valid tetrahedra");
			ImGui::Separator();
			ImGui::Text("Free boundary faces: %d", delaunay3d_free_boundary_faces);
			ImGui::Text("Total valid faces: %d", (int)global_all_valid_faces.size());
			ImGui::Separator();
			ImGui::Text("=== Manifold Diagnostic ===");
			if (global_manifold_info.is_manifold())
				ImGui::TextColored(ImVec4(0.0f, 0.6f, 0.0f, 1.0f), "Result: MANIFOLD");
			else
				ImGui::TextColored(ImVec4(0.8f, 0.0f, 0.0f, 1.0f), "Result: NOT MANIFOLD");
			ImGui::Text("Edge-manifold: %s", global_manifold_info.edge_manifold ? "YES" : "NO");
			ImGui::Text("Vertex-manifold: %s", global_manifold_info.vertex_manifold ? "YES" : "NO");
			ImGui::Text("Euler characteristic: %d", global_manifold_info.euler_characteristic);
			if (!global_manifold_info.vertex_manifold)
				ImGui::TextColored(ImVec4(0.8f, 0.0f, 0.0f, 1.0f), "Non-manifold vertices: %d", global_manifold_info.non_manifold_vertices);
			if (!global_manifold_info.edge_manifold)
				ImGui::TextColored(ImVec4(0.8f, 0.0f, 0.0f, 1.0f), "Non-manifold edges: %d", global_manifold_info.non_manifold_edges);
			ImGui::End();
		};
	}


	if (choice == 'N' || choice == 'n') {
		// =========================
		// VIEWER 3: CON INTERSECCIÃƒâ€œN
		// =========================
		Eigen::MatrixXd all_vertices(0, 3);
		Eigen::MatrixXi all_faces(0, 3);
		Eigen::MatrixXd all_colors(0, 3);

		int vertex_offset = 0;

		for (const auto& entry : global_similarity_matrix) {
			if (entry.faces.rows() > 0) {
				int n_verts = entry.vertices.rows();
				int n_faces = entry.faces.rows();

				all_vertices.conservativeResize(all_vertices.rows() + n_verts, 3);
				for (int i = 0; i < n_verts; ++i) {
					all_vertices.row(all_vertices.rows() - n_verts + i) = entry.vertices.row(i);
				}

				all_faces.conservativeResize(all_faces.rows() + n_faces, 3);
				for (int i = 0; i < n_faces; ++i) {
					all_faces.row(all_faces.rows() - n_faces + i) =
						entry.faces.row(i).array() + vertex_offset;
				}

				all_colors.conservativeResize(all_colors.rows() + n_faces, 3);
				for (int i = 0; i < n_faces; ++i) {
					all_colors.row(all_colors.rows() - n_faces + i) << 0.0, 0.5, 1.0;
				}

				vertex_offset += n_verts;
			}
		}

		if (all_faces.rows() > 0) {
			viewer_intersection.data().set_mesh(all_vertices, all_faces);
			viewer_intersection.data().set_colors(all_colors);
		}

		draw_contours_intersection(viewer_intersection, levels[startIdx - 1].contours,
			Eigen::RowVector3d(0, 0, 0));
		draw_contours_intersection(viewer_intersection, levels[startIdx].contours,
			Eigen::RowVector3d(1, 0, 0));

		// =========================
		// Ã¢Å“â€¦ VIEWER 4: SIN INTERSECCIÃƒâ€œN (SOLO SUPERPOSICIÃƒâ€œN)
		// =========================
		draw_contours_intersection(viewer_overlay, levels[startIdx - 1].contours,
			Eigen::RowVector3d(0, 0, 0));
		draw_contours_intersection(viewer_overlay, levels[startIdx].contours,
			Eigen::RowVector3d(1, 0, 0));

		viewer_overlay.data().set_mesh(Vbox, Fbox);
		viewer_overlay.data().show_faces = false;

		menu_inter.callback_draw_viewer_window = [&]() {

			// =========================
			// VENTANA 1: SIMILARITY MATRIX
			// =========================
			ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(1.0f, 1.0f, 1.0f, 1.0f));
			ImGui::PushStyleColor(ImGuiCol_ChildBg, ImVec4(0.95f, 0.95f, 0.95f, 1.0f));
			ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.0f, 0.0f, 0.0f, 1.0f));
			ImGui::PushStyleColor(ImGuiCol_TitleBg, ImVec4(0.85f, 0.85f, 0.85f, 1.0f));
			ImGui::PushStyleColor(ImGuiCol_TitleBgActive, ImVec4(0.75f, 0.75f, 0.75f, 1.0f));

			ImGui::SetNextWindowSize(ImVec2(500, 400), ImGuiCond_FirstUseEver);
			ImGui::Begin("Similarity Matrix");
			ImGui::SetWindowFontScale(1.6f);

			ImGui::BeginChild("SimilarityScroll", ImVec2(0, 0), true);

			for (const auto& entry : global_similarity_matrix) {
				std::string name_l1 = format_polygon_name_l1(entry.poly_l1_ref, all_polygons, forest_l1);
				std::string name_l2 = format_polygon_name_l2(entry.poly_l2_ref, all_polygons, forest_l2);

				if (entry.similarity_index < 1e-10) {
					ImGui::TextColored(ImVec4(0.5f, 0.5f, 0.5f, 1.0f),
						"%s vs %s = No intersection",
						name_l1.c_str(), name_l2.c_str());
				}
				else {
					ImGui::Text("%s vs %s = %.4f",
						name_l1.c_str(), name_l2.c_str(), entry.similarity_index);
				}
			}

			ImGui::EndChild();
			ImGui::SetWindowFontScale(1.0f);
			ImGui::End();

			ImGui::PopStyleColor(5);

			// =========================
// VENTANA 2: MAPPING GROUPS
// =========================
			ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(1.0f, 1.0f, 1.0f, 1.0f));
			ImGui::PushStyleColor(ImGuiCol_ChildBg, ImVec4(0.95f, 0.95f, 0.95f, 1.0f));
			ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.0f, 0.0f, 0.0f, 1.0f));
			ImGui::PushStyleColor(ImGuiCol_TitleBg, ImVec4(0.85f, 0.85f, 0.85f, 1.0f));
			ImGui::PushStyleColor(ImGuiCol_TitleBgActive, ImVec4(0.75f, 0.75f, 0.75f, 1.0f));

			ImGui::SetNextWindowSize(ImVec2(500, 300), ImGuiCond_FirstUseEver);
			ImGui::Begin("Mapping Groups");
			ImGui::SetWindowFontScale(1.6f);

			ImGui::BeginChild("MappingScroll", ImVec2(0, 0), true);

			// =========================
			// SOLID GROUPS (-s)
			// =========================
			for (size_t g = 0; g < global_mapping_groups.size(); ++g) {

				std::string l1 = "{}";
				std::string l2 = "{}";

				// Compute level automatically: minimum contour depth in the group
				int level = std::numeric_limits<int>::max();
				for (int pid : global_mapping_groups[g].polygons_l1)
					for (int cid : all_polygons[polygon_refs_l1[pid].polygon_id].contours)
						level = std::min(level, forest_l1[cid].depth);
				for (int pid : global_mapping_groups[g].polygons_l2)
					for (int cid : all_polygons[polygon_refs_l2[pid].polygon_id].contours)
						level = std::min(level, forest_l2[cid].depth);
				if (level == std::numeric_limits<int>::max()) level = -1;

				if (!global_mapping_groups[g].polygons_l1.empty()) {
					l1 = "{";
					for (size_t i = 0; i < global_mapping_groups[g].polygons_l1.size(); ++i) {
						int pid = global_mapping_groups[g].polygons_l1[i];
						l1 += format_polygon_name_l1(&polygon_refs_l1[pid], all_polygons, forest_l1);
						if (i + 1 < global_mapping_groups[g].polygons_l1.size())
							l1 += " ";
					}
					l1 += "}";
				}

				if (!global_mapping_groups[g].polygons_l2.empty()) {
					l2 = "{";
					for (size_t i = 0; i < global_mapping_groups[g].polygons_l2.size(); ++i) {
						int pid = global_mapping_groups[g].polygons_l2[i];
						l2 += format_polygon_name_l2(&polygon_refs_l2[pid], all_polygons, forest_l2);
						if (i + 1 < global_mapping_groups[g].polygons_l2.size())
							l2 += " ";
					}
					l2 += "}";
				}

				ImGui::Text("Solid: %zu. %s vs %s Level=%d",
					g + 1,
					l1.c_str(),
					l2.c_str(),
					level);
			}

			// =========================
			// HOLE GROUPS (-h)
			// =========================
			size_t base = global_mapping_groups.size();

			for (size_t g = 0; g < global_hole_mapping_groups.size(); ++g) {

				std::string l1 = "{}";
				std::string l2 = "{}";

				// Compute level automatically: minimum contour depth in the hole group
				int level = std::numeric_limits<int>::max();
				for (int pid : global_hole_mapping_groups[g].polygons_l1) {
					if (pid < (int)hole_refs_l1.size())
						for (int cid : all_hole_polygons[hole_refs_l1[pid].polygon_id].contours)
							level = std::min(level, forest_l1[cid].depth);
				}
				for (int pid : global_hole_mapping_groups[g].polygons_l2) {
					if (pid < (int)hole_refs_l2.size())
						for (int cid : all_hole_polygons[hole_refs_l2[pid].polygon_id].contours)
							level = std::min(level, forest_l2[cid].depth);
				}
				if (level == std::numeric_limits<int>::max()) level = -1;

				if (!global_hole_mapping_groups[g].polygons_l1.empty()) {
					l1 = "{";
					for (size_t i = 0; i < global_hole_mapping_groups[g].polygons_l1.size(); ++i) {
						int pid = global_hole_mapping_groups[g].polygons_l1[i];
						if (pid < (int)hole_refs_l1.size()) {
							l1 += format_polygon_name_l1(&hole_refs_l1[pid], all_hole_polygons, forest_l1);
							if (i + 1 < global_hole_mapping_groups[g].polygons_l1.size())
								l1 += " ";
						}
					}
					l1 += "}";
				}

				if (!global_hole_mapping_groups[g].polygons_l2.empty()) {
					l2 = "{";
					for (size_t i = 0; i < global_hole_mapping_groups[g].polygons_l2.size(); ++i) {
						int pid = global_hole_mapping_groups[g].polygons_l2[i];
						if (pid < (int)hole_refs_l2.size()) {
							l2 += format_polygon_name_l2(&hole_refs_l2[pid], all_hole_polygons, forest_l2);
							if (i + 1 < global_hole_mapping_groups[g].polygons_l2.size())
								l2 += " ";
						}
					}
					l2 += "}";
				}

				ImGui::Text("Hole: %zu. %s vs %s  Level=%d",
					base + g + 1,
					l1.c_str(),
					l2.c_str(),
					level);
			}

			ImGui::EndChild();
			ImGui::SetWindowFontScale(1.0f);
			ImGui::End();

			ImGui::PopStyleColor(5);

			// =========================
			// VENTANA 3: FINAL DEPURATED GROUPS (always when available)
			// =========================
			if (!global_final_groups_2d.empty()) {
				ImGui::PushStyleColor(ImGuiCol_WindowBg,      ImVec4(1.0f, 1.0f, 1.0f, 1.0f));
				ImGui::PushStyleColor(ImGuiCol_ChildBg,       ImVec4(0.95f, 0.95f, 0.95f, 1.0f));
				ImGui::PushStyleColor(ImGuiCol_Text,          ImVec4(0.0f, 0.0f, 0.0f, 1.0f));
				ImGui::PushStyleColor(ImGuiCol_TitleBg,       ImVec4(0.85f, 0.85f, 0.85f, 1.0f));
				ImGui::PushStyleColor(ImGuiCol_TitleBgActive, ImVec4(0.75f, 0.75f, 0.75f, 1.0f));

				ImGui::SetNextWindowSize(ImVec2(520, 350), ImGuiCond_FirstUseEver);
				ImGui::Begin("Final Depurated Groups");
				ImGui::SetWindowFontScale(1.5f);
				ImGui::Text("Depurated groups");
				ImGui::Separator();

				ImGui::BeginChild("FinalGroupsScroll", ImVec2(0, 0), true);
				for (int i = 0; i < (int)global_final_groups_2d.size(); ++i) {
					const auto& fg = global_final_groups_2d[i];

					// Build display strings
					std::string s_l1, s_l2;
					for (int pi = 0; pi < (int)fg.l1_side.size(); ++pi) {
						if (pi) s_l1 += " ";
						const auto& pp = fg.l1_side[pi];
						if (pp.contour_ids.size() == 1) {
							s_l1 += char('A' + forest_l1[pp.contour_ids[0]].contourId);
						} else {
							s_l1 += "(";
							for (int cid : pp.contour_ids) s_l1 += char('A' + forest_l1[cid].contourId);
							s_l1 += ")";
						}
					}
					for (int pi = 0; pi < (int)fg.l2_side.size(); ++pi) {
						if (pi) s_l2 += " ";
						const auto& pp = fg.l2_side[pi];
						if (pp.contour_ids.size() == 1) {
							s_l2 += std::to_string(forest_l2[pp.contour_ids[0]].contourId + 1);
						} else {
							s_l2 += "(";
							for (int cid : pp.contour_ids) s_l2 += std::to_string(forest_l2[cid].contourId + 1);
							s_l2 += ")";
						}
					}

					ImGui::Text("%d. {%s} vs {%s}",
						i + 1, s_l1.c_str(), s_l2.c_str());
				}
				ImGui::EndChild();
				ImGui::SetWindowFontScale(1.0f);
				ImGui::End();
				ImGui::PopStyleColor(5);
			}

			// =========================
			// LABELS sobre el viewer
			// =========================
			ImGui::SetWindowFontScale(1.8f);
			for (const auto& lbl : labels) {
				auto screen = projectToScreen(lbl.position, viewer_intersection);

				ImGui::GetForegroundDrawList()->AddText(
					ImVec2(screen.x(), screen.y()),
					IM_COL32(0, 0, 0, 255),
					lbl.text.c_str()
				);
			}

			for (const auto& lbl : labels_l2) {
				auto screen = projectToScreen(lbl.position, viewer_intersection);

				ImGui::GetForegroundDrawList()->AddText(
					ImVec2(screen.x(), screen.y()),
					IM_COL32(255, 0, 0, 255),
					lbl.text.c_str()
				);
			}
			ImGui::SetWindowFontScale(1.0f);
		};

		// Ã¢Å“â€¦ NUEVO: Callback para el Viewer 4 (SuperposiciÃƒÂ³n)
		menu_overlay.callback_draw_viewer_window = [&]() {
			ImGui::Begin("Overlay (L1 + L2)");

			ImGui::TextColored(ImVec4(0.0f, 0.0f, 0.0f, 1.0f), "Level L1: Black");
			ImGui::TextColored(ImVec4(1.0f, 0.0f, 0.0f, 1.0f), "Level L2: Red");

			ImGui::Separator();

			ImGui::Text("Forest L1: %s", global_forest_text_l1.c_str());
			ImGui::Text("Forest L2: %s", global_forest_text_l2.c_str());

			ImGui::End();

			ImGui::SetWindowFontScale(1.8f);
			for (const auto& lbl : labels) {
				auto screen = projectToScreen(lbl.position, viewer_overlay);

				ImGui::GetForegroundDrawList()->AddText(
					ImVec2(screen.x(), screen.y()),
					IM_COL32(0, 0, 0, 255),
					lbl.text.c_str()
				);
			}

			for (const auto& lbl : labels_l2) {
				auto screen = projectToScreen(lbl.position, viewer_overlay);

				ImGui::GetForegroundDrawList()->AddText(
					ImVec2(screen.x(), screen.y()),
					IM_COL32(255, 0, 0, 255),
					lbl.text.c_str()
				);
			}
			ImGui::SetWindowFontScale(1.0f);
		};

		// =========================
		// VIEWER 5: HOLE vs HOLE - DATA
		// =========================
		{
			Eigen::MatrixXd hh_verts(0, 3);
			Eigen::MatrixXi hh_faces(0, 3);
			Eigen::MatrixXd hh_colors(0, 3);
			int hh_offset = 0;

			for (const auto& entry : hole_vs_hole_matrix) {
				if (entry.faces.rows() > 0) {
					int nv = entry.vertices.rows();
					int nf = entry.faces.rows();
					hh_verts.conservativeResize(hh_verts.rows() + nv, 3);
					for (int i = 0; i < nv; ++i)
						hh_verts.row(hh_verts.rows() - nv + i) = entry.vertices.row(i);
					hh_faces.conservativeResize(hh_faces.rows() + nf, 3);
					for (int i = 0; i < nf; ++i)
						hh_faces.row(hh_faces.rows() - nf + i) = entry.faces.row(i).array() + hh_offset;
					hh_colors.conservativeResize(hh_colors.rows() + nf, 3);
					for (int i = 0; i < nf; ++i)
						hh_colors.row(hh_colors.rows() - nf + i) << 0.8, 0.2, 0.8;
					hh_offset += nv;
				}
			}

			if (hh_faces.rows() > 0) {
				viewer_hh.data().set_mesh(hh_verts, hh_faces);
				viewer_hh.data().set_colors(hh_colors);
			} else {
				viewer_hh.data().set_mesh(Vbox, Fbox);
				viewer_hh.data().show_faces = false;
			}

			draw_contours_intersection(viewer_hh, levels[startIdx - 1].contours,
				Eigen::RowVector3d(0, 0, 0));
			draw_contours_intersection(viewer_hh, levels[startIdx].contours,
				Eigen::RowVector3d(1, 0, 0));
		}

		// =========================
		// VIEWER 6: HOLE vs SOLID - DATA
		// =========================
		{
			Eigen::MatrixXd hs_verts(0, 3);
			Eigen::MatrixXi hs_faces(0, 3);
			Eigen::MatrixXd hs_colors(0, 3);
			int hs_offset = 0;

			for (const auto& entry : hole_vs_solid_matrix) {
				if (entry.faces.rows() > 0) {
					int nv = entry.vertices.rows();
					int nf = entry.faces.rows();
					hs_verts.conservativeResize(hs_verts.rows() + nv, 3);
					for (int i = 0; i < nv; ++i)
						hs_verts.row(hs_verts.rows() - nv + i) = entry.vertices.row(i);
					hs_faces.conservativeResize(hs_faces.rows() + nf, 3);
					for (int i = 0; i < nf; ++i)
						hs_faces.row(hs_faces.rows() - nf + i) = entry.faces.row(i).array() + hs_offset;
					hs_colors.conservativeResize(hs_colors.rows() + nf, 3);
					for (int i = 0; i < nf; ++i)
						hs_colors.row(hs_colors.rows() - nf + i) << 1.0, 0.5, 0.0;
					hs_offset += nv;
				}
			}

			if (hs_faces.rows() > 0) {
				viewer_hs.data().set_mesh(hs_verts, hs_faces);
				viewer_hs.data().set_colors(hs_colors);
			} else {
				viewer_hs.data().set_mesh(Vbox, Fbox);
				viewer_hs.data().show_faces = false;
			}

			draw_contours_intersection(viewer_hs, levels[startIdx - 1].contours,
				Eigen::RowVector3d(0, 0, 0));
			draw_contours_intersection(viewer_hs, levels[startIdx].contours,
				Eigen::RowVector3d(1, 0, 0));
		}

		// =========================
		// VIEWER 5: HOLE vs HOLE - CALLBACK
		// =========================
		menu_hh.callback_draw_viewer_window = [&]() {
			ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(1.0f, 1.0f, 1.0f, 1.0f));
			ImGui::PushStyleColor(ImGuiCol_ChildBg, ImVec4(0.95f, 0.95f, 0.95f, 1.0f));
			ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.0f, 0.0f, 0.0f, 1.0f));
			ImGui::PushStyleColor(ImGuiCol_TitleBg, ImVec4(0.85f, 0.85f, 0.85f, 1.0f));
			ImGui::PushStyleColor(ImGuiCol_TitleBgActive, ImVec4(0.75f, 0.75f, 0.75f, 1.0f));

			ImGui::SetNextWindowSize(ImVec2(550, 400), ImGuiCond_FirstUseEver);
			ImGui::Begin("Hole vs Hole Similarity");
			ImGui::SetWindowFontScale(1.6f);

			ImGui::BeginChild("HHScroll", ImVec2(0, 0), true);

			for (const auto& entry : hole_vs_hole_matrix) {
				std::string n1 = format_polygon_name_l1(entry.poly_l1_ref, all_hole_polygons, forest_l1);
				std::string n2 = format_polygon_name_l2(entry.poly_l2_ref, all_hole_polygons, forest_l2);

				if (entry.intersection_area < 1e-10) {
					ImGui::TextColored(ImVec4(0.5f, 0.5f, 0.5f, 1.0f),
						"%s vs %s = No intersection", n1.c_str(), n2.c_str());
				} else {
					double r1 = entry.intersection_area / entry.area_l1;
					double r2 = entry.intersection_area / entry.area_l2;
					ImGui::Text("%s vs %s  r1=%.4f  r2=%.4f",
						n1.c_str(), n2.c_str(), r1, r2);
				}
			}

			ImGui::EndChild();
			ImGui::SetWindowFontScale(1.0f);
			ImGui::End();
			ImGui::PopStyleColor(5);

			ImGui::SetWindowFontScale(1.8f);
			for (const auto& lbl : labels) {
				auto screen = projectToScreen(lbl.position, viewer_hh);
				ImGui::GetForegroundDrawList()->AddText(
					ImVec2(screen.x(), screen.y()), IM_COL32(0, 0, 0, 255), lbl.text.c_str());
			}
			for (const auto& lbl : labels_l2) {
				auto screen = projectToScreen(lbl.position, viewer_hh);
				ImGui::GetForegroundDrawList()->AddText(
					ImVec2(screen.x(), screen.y()), IM_COL32(255, 0, 0, 255), lbl.text.c_str());
			}
			ImGui::SetWindowFontScale(1.0f);
		};

		// =========================
		// VIEWER 6: HOLE vs SOLID - CALLBACK
		// =========================
		menu_hs.callback_draw_viewer_window = [&]() {
			ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(1.0f, 1.0f, 1.0f, 1.0f));
			ImGui::PushStyleColor(ImGuiCol_ChildBg, ImVec4(0.95f, 0.95f, 0.95f, 1.0f));
			ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.0f, 0.0f, 0.0f, 1.0f));
			ImGui::PushStyleColor(ImGuiCol_TitleBg, ImVec4(0.85f, 0.85f, 0.85f, 1.0f));
			ImGui::PushStyleColor(ImGuiCol_TitleBgActive, ImVec4(0.75f, 0.75f, 0.75f, 1.0f));

			ImGui::SetNextWindowSize(ImVec2(550, 400), ImGuiCond_FirstUseEver);
			ImGui::Begin("Hole vs Solid Similarity");
			ImGui::SetWindowFontScale(1.6f);

			ImGui::BeginChild("HSScroll", ImVec2(0, 0), true);

			for (const auto& entry : hole_vs_solid_matrix) {
				std::string n1, n2;
				if (entry.poly_l1_ref->level == 0) {
					n1 = "H" + format_polygon_name_l1(entry.poly_l1_ref, all_hole_polygons, forest_l1);
					n2 = "S" + format_polygon_name_l2(entry.poly_l2_ref, all_polygons, forest_l2);
				} else {
					n1 = "H" + format_polygon_name_l2(entry.poly_l1_ref, all_hole_polygons, forest_l2);
					n2 = "S" + format_polygon_name_l1(entry.poly_l2_ref, all_polygons, forest_l1);
				}

				if (entry.intersection_area < 1e-10) {
					ImGui::TextColored(ImVec4(0.5f, 0.5f, 0.5f, 1.0f),
						"%s vs %s = No intersection", n1.c_str(), n2.c_str());
				} else {
					double r1 = entry.intersection_area / entry.area_l1;
					double r2 = entry.intersection_area / entry.area_l2;
					ImGui::Text("%s vs %s  r1=%.4f  r2=%.4f",
						n1.c_str(), n2.c_str(), r1, r2);
				}
			}

			ImGui::EndChild();
			ImGui::SetWindowFontScale(1.0f);
			ImGui::End();
			ImGui::PopStyleColor(5);

			ImGui::SetWindowFontScale(1.8f);
			for (const auto& lbl : labels) {
				auto screen = projectToScreen(lbl.position, viewer_hs);
				ImGui::GetForegroundDrawList()->AddText(
					ImVec2(screen.x(), screen.y()), IM_COL32(0, 0, 0, 255), lbl.text.c_str());
			}
			for (const auto& lbl : labels_l2) {
				auto screen = projectToScreen(lbl.position, viewer_hs);
				ImGui::GetForegroundDrawList()->AddText(
					ImVec2(screen.x(), screen.y()), IM_COL32(255, 0, 0, 255), lbl.text.c_str());
			}
			ImGui::SetWindowFontScale(1.0f);
		};
	}

	// =========================
	// VIEWER 7 - VORONOI L1
	// =========================
	igl::opengl::glfw::Viewer viewer_vor_l1;
	igl::opengl::glfw::imgui::ImGuiPlugin plugin_vor_l1;
	igl::opengl::glfw::imgui::ImGuiMenu menu_vor_l1;

	viewer_vor_l1.plugins.push_back(&plugin_vor_l1);
	plugin_vor_l1.widgets.push_back(&menu_vor_l1);

	viewer_vor_l1.core().background_color << 1.0, 1.0, 1.0, 1.0;
	viewer_vor_l1.data().clear();
	viewer_vor_l1.data().line_width = 2.0f;

	// =========================
	// VIEWER 8 - VORONOI L2
	// =========================
	igl::opengl::glfw::Viewer viewer_vor_l2;
	igl::opengl::glfw::imgui::ImGuiPlugin plugin_vor_l2;
	igl::opengl::glfw::imgui::ImGuiMenu menu_vor_l2;

	viewer_vor_l2.plugins.push_back(&plugin_vor_l2);
	plugin_vor_l2.widgets.push_back(&menu_vor_l2);

	viewer_vor_l2.core().background_color << 1.0, 1.0, 1.0, 1.0;
	viewer_vor_l2.data().clear();
	viewer_vor_l2.data().line_width = 2.0f;

	// =========================
	// VIEWER 9 - IMA L1
	// =========================
	igl::opengl::glfw::Viewer viewer_ima_l1;
	igl::opengl::glfw::imgui::ImGuiPlugin plugin_ima_l1;
	igl::opengl::glfw::imgui::ImGuiMenu menu_ima_l1;

	viewer_ima_l1.plugins.push_back(&plugin_ima_l1);
	plugin_ima_l1.widgets.push_back(&menu_ima_l1);

	viewer_ima_l1.core().background_color << 1.0, 1.0, 1.0, 1.0;
	viewer_ima_l1.data().clear();
	viewer_ima_l1.data().line_width = 2.0f;

	// =========================
	// VIEWER 10 - EMA L1
	// =========================
	igl::opengl::glfw::Viewer viewer_ema_l1;
	igl::opengl::glfw::imgui::ImGuiPlugin plugin_ema_l1;
	igl::opengl::glfw::imgui::ImGuiMenu menu_ema_l1;

	viewer_ema_l1.plugins.push_back(&plugin_ema_l1);
	plugin_ema_l1.widgets.push_back(&menu_ema_l1);

	viewer_ema_l1.core().background_color << 1.0, 1.0, 1.0, 1.0;
	viewer_ema_l1.data().clear();
	viewer_ema_l1.data().line_width = 2.0f;

	// =========================
	// VIEWER 11 - IMA L2
	// =========================
	igl::opengl::glfw::Viewer viewer_ima_l2;
	igl::opengl::glfw::imgui::ImGuiPlugin plugin_ima_l2;
	igl::opengl::glfw::imgui::ImGuiMenu menu_ima_l2;

	viewer_ima_l2.plugins.push_back(&plugin_ima_l2);
	plugin_ima_l2.widgets.push_back(&menu_ima_l2);

	viewer_ima_l2.core().background_color << 1.0, 1.0, 1.0, 1.0;
	viewer_ima_l2.data().clear();
	viewer_ima_l2.data().line_width = 2.0f;

	// =========================
	// VIEWER 12 - EMA L2
	// =========================
	igl::opengl::glfw::Viewer viewer_ema_l2;
	igl::opengl::glfw::imgui::ImGuiPlugin plugin_ema_l2;
	igl::opengl::glfw::imgui::ImGuiMenu menu_ema_l2;

	viewer_ema_l2.plugins.push_back(&plugin_ema_l2);
	plugin_ema_l2.widgets.push_back(&menu_ema_l2);

	viewer_ema_l2.core().background_color << 1.0, 1.0, 1.0, 1.0;
	viewer_ema_l2.data().clear();
	viewer_ema_l2.data().line_width = 2.0f;

	// =========================
	// VIEWER 15 - EMA L1->L2 PROJECTION
	// =========================
	igl::opengl::glfw::Viewer viewer_proj_l1_on_l2;
	igl::opengl::glfw::imgui::ImGuiPlugin plugin_proj_l1_on_l2;
	igl::opengl::glfw::imgui::ImGuiMenu menu_proj_l1_on_l2;

	viewer_proj_l1_on_l2.plugins.push_back(&plugin_proj_l1_on_l2);
	plugin_proj_l1_on_l2.widgets.push_back(&menu_proj_l1_on_l2);

	viewer_proj_l1_on_l2.core().background_color << 1.0, 1.0, 1.0, 1.0;
	viewer_proj_l1_on_l2.data().clear();
	viewer_proj_l1_on_l2.data().line_width = 2.0f;

	// =========================
	// VIEWER 16 - EMA L2->L1 PROJECTION
	// =========================
	igl::opengl::glfw::Viewer viewer_proj_l2_on_l1;
	igl::opengl::glfw::imgui::ImGuiPlugin plugin_proj_l2_on_l1;
	igl::opengl::glfw::imgui::ImGuiMenu menu_proj_l2_on_l1;

	viewer_proj_l2_on_l1.plugins.push_back(&plugin_proj_l2_on_l1);
	plugin_proj_l2_on_l1.widgets.push_back(&menu_proj_l2_on_l1);

	viewer_proj_l2_on_l1.core().background_color << 1.0, 1.0, 1.0, 1.0;
	viewer_proj_l2_on_l1.data().clear();
	viewer_proj_l2_on_l1.data().line_width = 2.0f;

	// =========================
	// VIEWER 13 - IMA+EMA L1
	// =========================
	igl::opengl::glfw::Viewer viewer_both_l1;
	igl::opengl::glfw::imgui::ImGuiPlugin plugin_both_l1;
	igl::opengl::glfw::imgui::ImGuiMenu menu_both_l1;

	viewer_both_l1.plugins.push_back(&plugin_both_l1);
	plugin_both_l1.widgets.push_back(&menu_both_l1);

	viewer_both_l1.core().background_color << 1.0, 1.0, 1.0, 1.0;
	viewer_both_l1.data().clear();
	viewer_both_l1.data().line_width = 2.0f;

	// =========================
	// VIEWER 14 - IMA+EMA L2
	// =========================
	igl::opengl::glfw::Viewer viewer_both_l2;
	igl::opengl::glfw::imgui::ImGuiPlugin plugin_both_l2;
	igl::opengl::glfw::imgui::ImGuiMenu menu_both_l2;

	viewer_both_l2.plugins.push_back(&plugin_both_l2);
	plugin_both_l2.widgets.push_back(&menu_both_l2);

	viewer_both_l2.core().background_color << 1.0, 1.0, 1.0, 1.0;
	viewer_both_l2.data().clear();
	viewer_both_l2.data().line_width = 2.0f;

	// =========================
	// VIEWER 17 - 3D DELAUNAY
	// =========================
	igl::opengl::glfw::Viewer viewer_delaunay3d;
	igl::opengl::glfw::imgui::ImGuiPlugin plugin_delaunay3d;
	igl::opengl::glfw::imgui::ImGuiMenu menu_delaunay3d;

	viewer_delaunay3d.plugins.push_back(&plugin_delaunay3d);
	plugin_delaunay3d.widgets.push_back(&menu_delaunay3d);

	viewer_delaunay3d.core().background_color << 1.0, 1.0, 1.0, 1.0;
	viewer_delaunay3d.core().lighting_factor = 0.0f;
	viewer_delaunay3d.data().clear();
	viewer_delaunay3d.data().line_width = 2.0f;

	// =========================
	// VIEWER 18 - T1 TETRAHEDRA
	// =========================
	igl::opengl::glfw::Viewer viewer_t1;
	igl::opengl::glfw::imgui::ImGuiPlugin plugin_t1;
	igl::opengl::glfw::imgui::ImGuiMenu menu_t1;

	viewer_t1.plugins.push_back(&plugin_t1);
	plugin_t1.widgets.push_back(&menu_t1);

	viewer_t1.core().background_color << 1.0, 1.0, 1.0, 1.0;
	viewer_t1.core().lighting_factor = 0.0f;
	viewer_t1.data().clear();
	viewer_t1.data().line_width = 2.0f;

	// =========================
	// VIEWER 19 - T2 TETRAHEDRA
	// =========================
	igl::opengl::glfw::Viewer viewer_t2;
	igl::opengl::glfw::imgui::ImGuiPlugin plugin_t2;
	igl::opengl::glfw::imgui::ImGuiMenu menu_t2;

	viewer_t2.plugins.push_back(&plugin_t2);
	plugin_t2.widgets.push_back(&menu_t2);

	viewer_t2.core().background_color << 1.0, 1.0, 1.0, 1.0;
	viewer_t2.core().lighting_factor = 0.0f;
	viewer_t2.data().clear();
	viewer_t2.data().line_width = 2.0f;

	// =========================
	// VIEWER 20 - T12 TETRAHEDRA
	// =========================
	igl::opengl::glfw::Viewer viewer_t12;
	igl::opengl::glfw::imgui::ImGuiPlugin plugin_t12;
	igl::opengl::glfw::imgui::ImGuiMenu menu_t12;

	viewer_t12.plugins.push_back(&plugin_t12);
	plugin_t12.widgets.push_back(&menu_t12);

	viewer_t12.core().background_color << 1.0, 1.0, 1.0, 1.0;
	viewer_t12.core().lighting_factor = 0.0f;
	viewer_t12.data().clear();
	viewer_t12.data().line_width = 2.0f;

	// =========================
	// VIEWER 21 - T1 VALID (FILTERED)
	// =========================
	igl::opengl::glfw::Viewer viewer_t1_valid;
	igl::opengl::glfw::imgui::ImGuiPlugin plugin_t1_valid;
	igl::opengl::glfw::imgui::ImGuiMenu menu_t1_valid;

	viewer_t1_valid.plugins.push_back(&plugin_t1_valid);
	plugin_t1_valid.widgets.push_back(&menu_t1_valid);

	viewer_t1_valid.core().background_color << 1.0, 1.0, 1.0, 1.0;
	viewer_t1_valid.core().lighting_factor = 0.0f;
	viewer_t1_valid.data().clear();
	viewer_t1_valid.data().line_width = 2.0f;

	// =========================
	// VIEWER 22 - T1 REMOVED
	// =========================
	igl::opengl::glfw::Viewer viewer_t1_removed;
	igl::opengl::glfw::imgui::ImGuiPlugin plugin_t1_removed;
	igl::opengl::glfw::imgui::ImGuiMenu menu_t1_removed;

	viewer_t1_removed.plugins.push_back(&plugin_t1_removed);
	plugin_t1_removed.widgets.push_back(&menu_t1_removed);

	viewer_t1_removed.core().background_color << 1.0, 1.0, 1.0, 1.0;
	viewer_t1_removed.core().lighting_factor = 0.0f;
	viewer_t1_removed.data().clear();
	viewer_t1_removed.data().line_width = 2.0f;

	// =========================
	// VIEWER 23 - T2 VALID (FILTERED)
	// =========================
	igl::opengl::glfw::Viewer viewer_t2_valid;
	igl::opengl::glfw::imgui::ImGuiPlugin plugin_t2_valid;
	igl::opengl::glfw::imgui::ImGuiMenu menu_t2_valid;

	viewer_t2_valid.plugins.push_back(&plugin_t2_valid);
	plugin_t2_valid.widgets.push_back(&menu_t2_valid);

	viewer_t2_valid.core().background_color << 1.0, 1.0, 1.0, 1.0;
	viewer_t2_valid.core().lighting_factor = 0.0f;
	viewer_t2_valid.data().clear();
	viewer_t2_valid.data().line_width = 2.0f;

	// =========================
	// VIEWER 24 - T2 REMOVED
	// =========================
	igl::opengl::glfw::Viewer viewer_t2_removed;
	igl::opengl::glfw::imgui::ImGuiPlugin plugin_t2_removed;
	igl::opengl::glfw::imgui::ImGuiMenu menu_t2_removed;

	viewer_t2_removed.plugins.push_back(&plugin_t2_removed);
	plugin_t2_removed.widgets.push_back(&menu_t2_removed);

	viewer_t2_removed.core().background_color << 1.0, 1.0, 1.0, 1.0;
	viewer_t2_removed.core().lighting_factor = 0.0f;
	viewer_t2_removed.data().clear();
	viewer_t2_removed.data().line_width = 2.0f;

	// =========================
	// VIEWER 25 - ALL VALID (T1v + T2v + T12v)
	// =========================
	igl::opengl::glfw::Viewer viewer_all_valid;
	igl::opengl::glfw::imgui::ImGuiPlugin plugin_all_valid;
	igl::opengl::glfw::imgui::ImGuiMenu menu_all_valid;

	viewer_all_valid.plugins.push_back(&plugin_all_valid);
	plugin_all_valid.widgets.push_back(&menu_all_valid);

	viewer_all_valid.core().background_color << 1.0, 1.0, 1.0, 1.0;
	viewer_all_valid.core().lighting_factor = 0.0f;
	viewer_all_valid.data().clear();
	viewer_all_valid.data().line_width = 2.0f;

	// =========================
	// VIEWER 30 - ALL VALID GRAY (uniform gray)
	// =========================
	igl::opengl::glfw::Viewer viewer_all_valid_gray;
	igl::opengl::glfw::imgui::ImGuiPlugin plugin_all_valid_gray;
	igl::opengl::glfw::imgui::ImGuiMenu menu_all_valid_gray;

	viewer_all_valid_gray.plugins.push_back(&plugin_all_valid_gray);
	plugin_all_valid_gray.widgets.push_back(&menu_all_valid_gray);

	viewer_all_valid_gray.core().background_color << 1.0, 1.0, 1.0, 1.0;
	viewer_all_valid_gray.core().lighting_factor = 0.0f;
	viewer_all_valid_gray.data().clear();
	viewer_all_valid_gray.data().line_width = 2.0f;

	// =========================
	// VIEWER 31 - FREE BOUNDARY (skin surface)
	// =========================
	igl::opengl::glfw::Viewer viewer_free_boundary;
	igl::opengl::glfw::imgui::ImGuiPlugin plugin_free_boundary;
	igl::opengl::glfw::imgui::ImGuiMenu menu_free_boundary;

	viewer_free_boundary.plugins.push_back(&plugin_free_boundary);
	plugin_free_boundary.widgets.push_back(&menu_free_boundary);

	viewer_free_boundary.core().background_color << 1.0, 1.0, 1.0, 1.0;
	viewer_free_boundary.core().lighting_factor = 0.0f;
	viewer_free_boundary.data().clear();
	viewer_free_boundary.data().line_width = 2.0f;

	// =========================
	// VIEWER 28 - INTERACTIVE VALID (toggle T1v/T2v/T12v)
	// =========================
	igl::opengl::glfw::Viewer viewer_interactive;
	igl::opengl::glfw::imgui::ImGuiPlugin plugin_interactive;
	igl::opengl::glfw::imgui::ImGuiMenu menu_interactive;

	viewer_interactive.plugins.push_back(&plugin_interactive);
	plugin_interactive.widgets.push_back(&menu_interactive);

	viewer_interactive.core().background_color << 1.0, 1.0, 1.0, 1.0;
	viewer_interactive.core().lighting_factor = 0.0f;
	viewer_interactive.data().clear();
	viewer_interactive.data().line_width = 2.0f;

	// =========================
	// VIEWER 29 - T12 INSPECT (selected T12 + T1/T2 neighbors)
	// =========================
	igl::opengl::glfw::Viewer viewer_t12_inspect;
	igl::opengl::glfw::imgui::ImGuiPlugin plugin_t12_inspect;
	igl::opengl::glfw::imgui::ImGuiMenu menu_t12_inspect;

	viewer_t12_inspect.plugins.push_back(&plugin_t12_inspect);
	plugin_t12_inspect.widgets.push_back(&menu_t12_inspect);

	viewer_t12_inspect.core().background_color << 1.0, 1.0, 1.0, 1.0;
	viewer_t12_inspect.core().lighting_factor = 0.0f;
	viewer_t12_inspect.data().clear();
	viewer_t12_inspect.data().line_width = 2.0f;

	// =========================
	// VIEWER 32 - OBJ RESULT (read back exported OBJ)
	// =========================
	igl::opengl::glfw::Viewer viewer_obj_result;
	igl::opengl::glfw::imgui::ImGuiPlugin plugin_obj_result;
	igl::opengl::glfw::imgui::ImGuiMenu menu_obj_result;

	viewer_obj_result.plugins.push_back(&plugin_obj_result);
	plugin_obj_result.widgets.push_back(&menu_obj_result);

	viewer_obj_result.core().background_color << 1.0, 1.0, 1.0, 1.0;
	viewer_obj_result.data().clear();
	viewer_obj_result.data().line_width = 1.0f;

	menu_obj_result.callback_draw_viewer_window = [&]() {
		ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(1.0f, 1.0f, 1.0f, 1.0f));
		ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.0f, 0.0f, 0.0f, 1.0f));
		ImGui::PushStyleColor(ImGuiCol_TitleBg, ImVec4(0.85f, 0.85f, 0.85f, 1.0f));
		ImGui::PushStyleColor(ImGuiCol_TitleBgActive, ImVec4(0.75f, 0.75f, 0.75f, 1.0f));

		ImGui::Begin("OBJ Result");
		ImGui::SetWindowFontScale(1.4f);
		ImGui::Text("Loaded from: free_boundary.obj");
		ImGui::Text("Faces oriented outward (centroid-based)");

		ImGui::SetWindowFontScale(1.0f);
		ImGui::End();

		ImGui::PopStyleColor(4);
	};


	// =========================
	// VIEWER 26 - T12 VALID
	// =========================
	igl::opengl::glfw::Viewer viewer_t12_valid;
	igl::opengl::glfw::imgui::ImGuiPlugin plugin_t12_valid;
	igl::opengl::glfw::imgui::ImGuiMenu menu_t12_valid;

	viewer_t12_valid.plugins.push_back(&plugin_t12_valid);
	plugin_t12_valid.widgets.push_back(&menu_t12_valid);

	viewer_t12_valid.core().background_color << 1.0, 1.0, 1.0, 1.0;
	viewer_t12_valid.core().lighting_factor = 0.0f;
	viewer_t12_valid.data().clear();
	viewer_t12_valid.data().line_width = 2.0f;

	// =========================
	// VIEWER 27 - T12 REMOVED
	// =========================
	igl::opengl::glfw::Viewer viewer_t12_removed;
	igl::opengl::glfw::imgui::ImGuiPlugin plugin_t12_removed;
	igl::opengl::glfw::imgui::ImGuiMenu menu_t12_removed;

	viewer_t12_removed.plugins.push_back(&plugin_t12_removed);
	plugin_t12_removed.widgets.push_back(&menu_t12_removed);

	viewer_t12_removed.core().background_color << 1.0, 1.0, 1.0, 1.0;
	viewer_t12_removed.core().lighting_factor = 0.0f;
	viewer_t12_removed.data().clear();
	viewer_t12_removed.data().line_width = 2.0f;

	int delaunay3d_num_points = 0, delaunay3d_num_tets = 0, delaunay3d_num_faces = 0;
	int delaunay3d_T1 = 0, delaunay3d_T2 = 0, delaunay3d_T12 = 0, delaunay3d_degenerate = 0;
	int delaunay3d_T1_faces = 0, delaunay3d_T2_faces = 0, delaunay3d_T12_faces = 0;


	int ima_l1_count = 0, ema_l1_count = 0, disc_l1_count = 0;
	int ima_l2_count = 0, ema_l2_count = 0, disc_l2_count = 0;

	if (choice == 'N' || choice == 'n') {

		double z_l1 = levels[startIdx - 1].zCoord;
		double z_l2 = levels[startIdx].zCoord;

		// =========================
		// ALGORITHM 3: PER-GROUP RECONSTRUCTION
		// =========================
		if (false) { // algo 3 per-group Delaunay replaced: now uses single Delaunay + same filters as algo 4 with 2D similarity affinity groups
			std::cout << "\n=================================\n";
			std::cout << "ALGORITHM 3: PER-GROUP RECONSTRUCTION\n";
			std::cout << "=================================\n";

			const int fv3[4][3] = {{1,2,3},{0,2,3},{0,1,3},{0,1,2}};

			std::vector<Eigen::Vector3d> all_pts_3;
			std::vector<std::array<index_t,3>> a3_t1v, a3_t2v, a3_t12v;
			std::vector<std::array<index_t,3>> a3_t1v_rem, a3_t2v_rem, a3_t12v_rem;
			std::vector<std::array<index_t,3>> a3_t1_all, a3_t2_all, a3_t12_all;
			global_t1_tets.clear();
			global_t2_tets.clear();
			global_t12_tets.clear();
			index_t voffset = 0;

			for (int gi = 0; gi < (int)global_final_groups_2d.size(); ++gi) {
				const auto& fg = global_final_groups_2d[gi];
				std::cout << "\n--- Group " << (gi+1) << "/" << global_final_groups_2d.size() << " ---\n";

				// Build local contour arrays (LOCAL index 0..N-1)
				std::vector<datapoints> gc_l1, gc_l2;
				for (const auto& pp : fg.l1_side)
					for (int cid : pp.contour_ids)
						gc_l1.push_back(levels[startIdx-1].contours[cid]);
				for (const auto& pp : fg.l2_side)
					for (int cid : pp.contour_ids)
						gc_l2.push_back(levels[startIdx].contours[cid]);

				if (gc_l1.empty() || gc_l2.empty()) { std::cout << "  Skipped (empty side)\n"; continue; }

				// Build solidpolygon structs with LOCAL contour indices
				std::vector<solidpolygon> gp_l1, gp_l2;
				std::vector<polygon_reference> gr_l1, gr_l2;
				{
					int lc = 0;
					for (const auto& pp : fg.l1_side) {
						solidpolygon P;
						for (int k = 0; k < (int)pp.contour_ids.size(); ++k) P.contours.push_back(lc + k);
						gp_l1.push_back(P);
						gr_l1.push_back(polygon_reference((int)gp_l1.size()-1, 0));
						lc += (int)pp.contour_ids.size();
					}
					lc = 0;
					for (const auto& pp : fg.l2_side) {
						solidpolygon P;
						for (int k = 0; k < (int)pp.contour_ids.size(); ++k) P.contours.push_back(lc + k);
						gp_l2.push_back(P);
						gr_l2.push_back(polygon_reference((int)gp_l2.size()-1, 1));
						lc += (int)pp.contour_ids.size();
					}
				}

				SolidRegionGrid gg_l1, gg_l2;
				gg_l1.precompute(gr_l1, gp_l1, gc_l1);
				gg_l2.precompute(gr_l2, gp_l2, gc_l2);

				auto gr_vor_l1 = compute_voronoi_edges(resample_contours(gc_l1, resampleFactor));
				auto gma_l1 = classify_voronoi_edges(gr_vor_l1, gr_l1, gp_l1, gc_l1, &gg_l1);
				auto gr_vor_l2 = compute_voronoi_edges(resample_contours(gc_l2, resampleFactor));
				auto gma_l2 = classify_voronoi_edges(gr_vor_l2, gr_l2, gp_l2, gc_l2, &gg_l2);

				auto gproj_l2_on_l1 = compute_ema_projection(gma_l2.ema_edges, gr_l1, gp_l1, gc_l1, &gg_l1);
				auto gproj_l1_on_l2 = compute_ema_projection(gma_l1.ema_edges, gr_l2, gp_l2, gc_l2, &gg_l2);

				std::cout << "  IMA_L1=" << gma_l1.ima_edges.size()
					<< " IMA_L2=" << gma_l2.ima_edges.size()
					<< " EMA_L2->L1=" << gproj_l2_on_l1.solid_vertices.size()
					<< " EMA_L1->L2=" << gproj_l1_on_l2.solid_vertices.size() << "\n";

				// Augmented points (LOCAL z from levels)
				std::vector<Eigen::Vector3d> gaug_l1, gaug_l2;
				for (const auto& c : resample_contours(gc_l1, resampleFactor)) for (const auto& p : c.points) gaug_l1.push_back(p);
				index_t g_contour_l1 = (index_t)gaug_l1.size(); // contour pts count before IMA
				for (const auto& e : gma_l1.ima_edges) { gaug_l1.emplace_back(e.first.x,e.first.y,z_l1); gaug_l1.emplace_back(e.second.x,e.second.y,z_l1); }
				index_t g_ima_end_l1 = (index_t)gaug_l1.size(); // end of IMA in L1
				for (const auto& v : gproj_l2_on_l1.solid_vertices) gaug_l1.emplace_back(v.x(),v.y(),z_l1);
				for (const auto& c : resample_contours(gc_l2, resampleFactor)) for (const auto& p : c.points) gaug_l2.push_back(p);
				index_t g_contour_l2 = (index_t)gaug_l2.size(); // contour pts count before IMA
				for (const auto& e : gma_l2.ima_edges) { gaug_l2.emplace_back(e.first.x,e.first.y,z_l2); gaug_l2.emplace_back(e.second.x,e.second.y,z_l2); }
				index_t g_ima_end_l2 = (index_t)gaug_l2.size(); // end of IMA in L2
				for (const auto& v : gproj_l1_on_l2.solid_vertices) gaug_l2.emplace_back(v.x(),v.y(),z_l2);

				index_t n1g=(index_t)gaug_l1.size(), n2g=(index_t)gaug_l2.size(), N3dg=n1g+n2g;
				std::cout << "  Aug L1=" << n1g << " L2=" << n2g << " total=" << N3dg << "\n";
				if (N3dg < 4) { std::cout << "  Skipped (too few pts)\n"; continue; }

				std::vector<double> cg; cg.reserve(N3dg*3);
				for (const auto& p:gaug_l1){cg.push_back(p.x());cg.push_back(p.y());cg.push_back(p.z());}
				for (const auto& p:gaug_l2){cg.push_back(p.x());cg.push_back(p.y());cg.push_back(p.z());}

				Delaunay_var del3dg = Delaunay::create(3, "BDEL");
				del3dg->set_vertices(N3dg, cg.data());
				index_t nb_tg = del3dg->nb_cells();
				std::cout << "  Tets=" << nb_tg << "\n";

				std::vector<t1_t2_tet_info> gt1, gt2;
				std::vector<t12_tet_info> gt12;
				std::vector<std::array<index_t,3>> gfv_t1, gfv_t2;
				std::vector<std::array<index_t,3>> gfv_t1_rem, gfv_t2_rem;
				std::vector<std::array<index_t,3>> gfv_t1_all, gfv_t2_all, gfv_t12_rem, gfv_t12_all;

				for (index_t t=0; t<nb_tg; ++t) {
					const signed_index_t* v = del3dg->cell_to_v() + 4*t;
					int cnt1=0;
					for (int lv=0;lv<4;++lv) if((index_t)v[lv]<n1g) cnt1++;

					if (cnt1==3 || cnt1==1) {
						bool in_l1=(cnt1==3);
						t1_t2_tet_info info; info.type=in_l1?1:2;
						int bc=0; info.apex_vert=0;
						for (int lv=0;lv<4;++lv) {
							info.orig_verts[lv]=(index_t)v[lv];
							bool is_base=in_l1?((index_t)v[lv]<n1g):((index_t)v[lv]>=n1g);
							if(is_base) info.base_verts[bc++]=(index_t)v[lv]; else info.apex_vert=(index_t)v[lv];
						}
						for (int lf=0;lf<4;++lf)
							info.faces[lf]={voffset+(index_t)v[fv3[lf][0]],voffset+(index_t)v[fv3[lf][1]],voffset+(index_t)v[fv3[lf][2]]};
						info.centroid_x=(cg[3*info.base_verts[0]]+cg[3*info.base_verts[1]]+cg[3*info.base_verts[2]])/3.0;
						info.centroid_y=(cg[3*info.base_verts[0]+1]+cg[3*info.base_verts[1]+1]+cg[3*info.base_verts[2]+1])/3.0;
						const SolidRegionGrid& bg=in_l1?gg_l1:gg_l2;
						info.centroid_in_solid=(bg.classify(info.centroid_x,info.centroid_y)>=0);
						info.fails_midpoint=false; info.void_l1_count=0; info.void_l2_count=0;
						{
							const double st[3]={2./5.,0.5,3./5.}; bool av1=true,av2=true;
							for (int be=0;be<3;++be) {
								double bx=cg[3*info.base_verts[be]],by=cg[3*info.base_verts[be]+1];
								double ax=cg[3*info.apex_vert],ay=cg[3*info.apex_vert+1];
								for (int si=0;si<3;++si) {
									double mx=bx+st[si]*(ax-bx),my=by+st[si]*(ay-by);
									info.midpoint_x[be][si]=mx; info.midpoint_y[be][si]=my;
									info.midpoint_in_solid_l1[be][si]=(gg_l1.classify(mx,my)>=0);
									info.midpoint_in_solid_l2[be][si]=(gg_l2.classify(mx,my)>=0);
									if(!info.midpoint_in_solid_l1[be][si]) info.void_l1_count++;
									if(!info.midpoint_in_solid_l2[be][si]) info.void_l2_count++;
									if(info.midpoint_in_solid_l1[be][si]) av1=false;
									if(info.midpoint_in_solid_l2[be][si]) av2=false;
								}
							}
							if(av1&&av2) info.fails_midpoint=true;
						}
						if(in_l1) gt1.push_back(info); else gt2.push_back(info);
					}
					else if (cnt1==2) {
						index_t vl1[2]={},vl2[2]={}; int cl1=0,cl2=0;
						for (int lv=0;lv<4;++lv) {
							if((index_t)v[lv]<n1g){if(cl1<2)vl1[cl1++]=(index_t)v[lv];}
							else{if(cl2<2)vl2[cl2++]=(index_t)v[lv];}
						}
						t12_tet_info info; info.fails_midpoint=false; info.fails_isolated=false;
						for (int lv=0;lv<4;++lv) info.verts[lv]=voffset+(index_t)v[lv];
						for (int lf=0;lf<4;++lf)
							info.faces[lf]={voffset+(index_t)v[fv3[lf][0]],voffset+(index_t)v[fv3[lf][1]],voffset+(index_t)v[fv3[lf][2]]};
						{
							double mx1=(cg[3*vl1[0]]+cg[3*vl1[1]])*0.5,my1=(cg[3*vl1[0]+1]+cg[3*vl1[1]+1])*0.5;
							Eigen::Vector2d m1(mx1,my1);
							int r1=gg_l1.classify(mx1,my1);
							if(r1<0&&point_near_contour_boundary(m1,gr_l1,gp_l1,gc_l1)) r1=0;
							if(r1<0){info.fails_midpoint=true;}
							else{
								double mx2=(cg[3*vl2[0]]+cg[3*vl2[1]])*0.5,my2=(cg[3*vl2[0]+1]+cg[3*vl2[1]+1])*0.5;
								Eigen::Vector2d m2(mx2,my2);
								int r2=gg_l2.classify(mx2,my2);
								if(r2<0&&point_near_contour_boundary(m2,gr_l2,gp_l2,gc_l2)) r2=0;
								if(r2<0) info.fails_midpoint=true;
							}
						}
						gt12.push_back(info);
					}
				}

				// T1/T2 valid + removed faces (global indices via voffset)
				for (const auto& info:gt1) {
					bool valid = info.centroid_in_solid && !(global_filter_t1_midpoint && info.fails_midpoint);
					for(int lf=0;lf<4;++lf) gfv_t1_all.push_back(info.faces[lf]);
					if(valid) for(int lf=0;lf<4;++lf) gfv_t1.push_back(info.faces[lf]);
					else      for(int lf=0;lf<4;++lf) gfv_t1_rem.push_back(info.faces[lf]);
				}
				for (const auto& info:gt2) {
					bool valid = info.centroid_in_solid && !(global_filter_t2_midpoint && info.fails_midpoint);
					for(int lf=0;lf<4;++lf) gfv_t2_all.push_back(info.faces[lf]);
					if(valid) for(int lf=0;lf<4;++lf) gfv_t2.push_back(info.faces[lf]);
					else      for(int lf=0;lf<4;++lf) gfv_t2_rem.push_back(info.faces[lf]);
				}

				// Isolated filter for T12
				{
					auto mk=[](index_t a,index_t b,index_t c)->std::array<index_t,3>{
						std::array<index_t,3> f={a,b,c};
						if(f[0]>f[1]) std::swap(f[0],f[1]);
						if(f[1]>f[2]) std::swap(f[1],f[2]);
						if(f[0]>f[1]) std::swap(f[0],f[1]); return f;};
					struct AH{size_t operator()(const std::array<index_t,3>&a)const{
						size_t h=std::hash<index_t>{}(a[0]);
						h^=std::hash<index_t>{}(a[1])+0x9e3779b9+(h<<6)+(h>>2);
						h^=std::hash<index_t>{}(a[2])+0x9e3779b9+(h<<6)+(h>>2); return h;}};
					std::unordered_set<std::array<index_t,3>,AH> rf;
					for(const auto& f:gfv_t1) rf.insert(mk(f[0],f[1],f[2]));
					for(const auto& f:gfv_t2) rf.insert(mk(f[0],f[1],f[2]));
					size_t n12=gt12.size();
					std::vector<bool> reach(n12,false);
					bool fc=true;
					while(fc){fc=false;
						for(size_t i=0;i<n12;++i){
							if(gt12[i].fails_midpoint||reach[i]) continue;
							for(int lf=0;lf<4;++lf)
								if(rf.count(mk(gt12[i].faces[lf][0],gt12[i].faces[lf][1],gt12[i].faces[lf][2]))){
									reach[i]=true;
									for(int lf2=0;lf2<4;++lf2) rf.insert(mk(gt12[i].faces[lf2][0],gt12[i].faces[lf2][1],gt12[i].faces[lf2][2]));
									fc=true; break;}}}
					for(size_t i=0;i<n12;++i) if(!reach[i]) gt12[i].fails_isolated=true;

					// IMA rescue (same as algo 4 inline): isolated T12s with an IMA vertex
					// whose segment midpoint is inside solid on both levels are rescued.
					// Local IMA index ranges use voffset for global addressing.
					if (global_filter_t12_ima_rescue) {
						index_t g_ima_start_l1_g = voffset + g_contour_l1;
						index_t g_ima_end_l1_g   = voffset + g_ima_end_l1;
						index_t g_ima_start_l2_g = voffset + n1g + g_contour_l2;
						index_t g_ima_end_l2_g   = voffset + n1g + g_ima_end_l2;
						int rescued = 0;
						for (size_t i = 0; i < n12; ++i) {
							if (!gt12[i].fails_isolated) continue;
							if (gt12[i].fails_midpoint) continue;
							// Separate verts into L1 (< voffset+n1g) and L2 pairs
							index_t vl1[2], vl2[2]; int cl1=0, cl2=0;
							for (int lv=0;lv<4;++lv) {
								index_t vi = gt12[i].faces[0][0]; // use verts array
								vi = gt12[i].verts[lv];
								if (vi >= voffset && vi < voffset + n1g) { if(cl1<2) vl1[cl1++]=vi; }
								else { if(cl2<2) vl2[cl2++]=vi; }
							}
							if (cl1!=2||cl2!=2) continue;
							bool l1_ima = (vl1[0]>=g_ima_start_l1_g&&vl1[0]<g_ima_end_l1_g)||
							             (vl1[1]>=g_ima_start_l1_g&&vl1[1]<g_ima_end_l1_g);
							bool l2_ima = (vl2[0]>=g_ima_start_l2_g&&vl2[0]<g_ima_end_l2_g)||
							             (vl2[1]>=g_ima_start_l2_g&&vl2[1]<g_ima_end_l2_g);
							if (!l1_ima && !l2_ima) continue;
							bool rescue = false;
							if (l1_ima) {
								double mx=(cg[3*(vl1[0]-voffset)]+cg[3*(vl1[1]-voffset)])*0.5;
								double my=(cg[3*(vl1[0]-voffset)+1]+cg[3*(vl1[1]-voffset)+1])*0.5;
								if (gg_l1.classify(mx,my)>=0 && gg_l2.classify(mx,my)>=0) rescue=true;
							}
							if (!rescue && l2_ima) {
								index_t vl2_local[2]={vl2[0]-voffset-n1g, vl2[1]-voffset-n1g};
								double mx=(cg[3*vl2_local[0]]+cg[3*vl2_local[1]])*0.5;
								double my=(cg[3*vl2_local[0]+1]+cg[3*vl2_local[1]+1])*0.5;
								if (gg_l1.classify(mx,my)>=0 && gg_l2.classify(mx,my)>=0) rescue=true;
							}
							if (rescue) { gt12[i].fails_isolated=false; rescued++; }
						}
						if (rescued) std::cout << "  IMA rescue: " << rescued << " T12s rescued\n";
					}
				}

				// Collect T12 valid + removed + all
				std::vector<std::array<index_t,3>> gfv_t12;
				for (const auto& info:gt12) {
					for(int lf=0;lf<4;++lf) gfv_t12_all.push_back(info.faces[lf]);
					if(!info.fails_midpoint&&!info.fails_isolated)
						for(int lf=0;lf<4;++lf) gfv_t12.push_back(info.faces[lf]);
					else
						for(int lf=0;lf<4;++lf) gfv_t12_rem.push_back(info.faces[lf]);
				}

				std::cout << "  T1v=" << (int)gfv_t1.size()/4
					<< " T2v=" << (int)gfv_t2.size()/4
					<< " T12v=" << (int)gfv_t12.size()/4 << "\n";

				// Accumulate into global arrays (faces already carry global indices via voffset)
				a3_t1v.insert(a3_t1v.end(),gfv_t1.begin(),gfv_t1.end());
				a3_t2v.insert(a3_t2v.end(),gfv_t2.begin(),gfv_t2.end());
				a3_t12v.insert(a3_t12v.end(),gfv_t12.begin(),gfv_t12.end());
				a3_t1v_rem.insert(a3_t1v_rem.end(),gfv_t1_rem.begin(),gfv_t1_rem.end());
				a3_t2v_rem.insert(a3_t2v_rem.end(),gfv_t2_rem.begin(),gfv_t2_rem.end());
				a3_t12v_rem.insert(a3_t12v_rem.end(),gfv_t12_rem.begin(),gfv_t12_rem.end());
				a3_t1_all.insert(a3_t1_all.end(),gfv_t1_all.begin(),gfv_t1_all.end());
				a3_t2_all.insert(a3_t2_all.end(),gfv_t2_all.begin(),gfv_t2_all.end());
				a3_t12_all.insert(a3_t12_all.end(),gfv_t12_all.begin(),gfv_t12_all.end());
				for(auto& info:gt1) global_t1_tets.push_back(info);
				for(auto& info:gt2) global_t2_tets.push_back(info);
				for(auto& info:gt12) global_t12_tets.push_back(info);
				for(const auto& p:gaug_l1) all_pts_3.push_back(p);
				for(const auto& p:gaug_l2) all_pts_3.push_back(p);
				voffset += N3dg;
			}

			// Build combined vertex matrix
			index_t Ntotal=(index_t)all_pts_3.size();
			Eigen::MatrixXd V3d_a3(Ntotal,3);
			for(index_t i=0;i<Ntotal;++i) V3d_a3.row(i)<<all_pts_3[i].x(),all_pts_3[i].y(),all_pts_3[i].z();

			global_V3d=V3d_a3;
			// Seed T1/T2 face lists before calling recompute functions
			global_faces_t1_valid=a3_t1v; global_faces_t2_valid=a3_t2v;
			global_faces_t1_removed=a3_t1v_rem; global_faces_t2_removed=a3_t2v_rem;
			delaunay3d_T1=(int)a3_t1_all.size()/4; delaunay3d_T1_faces=(int)a3_t1_all.size();
			delaunay3d_T2=(int)a3_t2_all.size()/4; delaunay3d_T2_faces=(int)a3_t2_all.size();
			delaunay3d_T12=(int)a3_t12_all.size()/4; delaunay3d_T12_faces=(int)a3_t12_all.size();
			delaunay3d_num_points=(int)Ntotal;
			delaunay3d_num_tets=delaunay3d_T1+delaunay3d_T2+delaunay3d_T12;
			interactive_needs_rebuild=true;
			// Algo 3 is multi-group: there is no single combined grid for IMA rescue,
			// so disable it by clearing the global grid/contour pointers and IMA ranges.
			// The isolated flood-fill in recompute_t12_from_filters works correctly
			// with the accumulated global T1/T2 valid face sets from all groups.
			global_n_l1_pts = 0;
			global_filter_contours_l1 = nullptr;
			global_filter_contours_l2 = nullptr;
			global_grid_l1 = nullptr;
			global_grid_l2 = nullptr;
			global_ima_start_l1 = 0; global_ima_end_l1 = 0;
			global_ima_start_l2 = 0; global_ima_end_l2 = 0;
			// Recompute T1/T2 valid/removed and their counts so interactive filter
			// toggles work correctly from the first frame (same conditions as per-group loop)
			recompute_t1_t2_from_filters();
			// Recompute T12 using the global isolated filter so that the interactive
			// viewer filter toggles behave correctly from the first frame
			recompute_t12_from_filters();
			delaunay3d_all_valid_faces=(int)(global_faces_t1_valid.size()+global_faces_t2_valid.size()+global_faces_t12_valid.size());

			std::cout << "\n=== ALGO3 TOTALS: V=" << Ntotal
				<< " T1v=" << delaunay3d_T1_valid_faces
				<< " T2v=" << delaunay3d_T2_valid_faces
				<< " T12v=" << delaunay3d_T12_valid_faces
				<< " all=" << delaunay3d_all_valid_faces << " ===\n";

			// Viewer setup helper
			auto set_tet_vw=[&](igl::opengl::glfw::Viewer& vw,
				const std::vector<std::array<index_t,3>>& faces,
				const Eigen::RowVector3d& color){
				if(!faces.empty()){
					int nf=(int)faces.size();
					Eigen::MatrixXi F(nf,3);
					for(int i=0;i<nf;++i) F.row(i)<<(int)faces[i][0],(int)faces[i][1],(int)faces[i][2];
					vw.data().set_mesh(V3d_a3,F);
					vw.data().show_lines=true; vw.data().show_faces=true;
					Eigen::MatrixXd C(nf,3); for(int i=0;i<nf;++i) C.row(i)=color;
					vw.data().set_colors(C);
				}else{vw.data().set_mesh(Vbox,Fbox); vw.data().show_faces=false;}};

			set_tet_vw(viewer_t1_valid,  global_faces_t1_valid,   Eigen::RowVector3d(0.4,0.6,1.0));
			set_tet_vw(viewer_t2_valid,  global_faces_t2_valid,   Eigen::RowVector3d(0.4,1.0,0.4));
			set_tet_vw(viewer_t12_valid, global_faces_t12_valid,  Eigen::RowVector3d(1.0,0.7,0.3));
			set_tet_vw(viewer_t1,        a3_t1_all,  Eigen::RowVector3d(0.4,0.6,1.0));
			set_tet_vw(viewer_t2,        a3_t2_all,  Eigen::RowVector3d(0.4,1.0,0.4));
			set_tet_vw(viewer_t12,       a3_t12_all, Eigen::RowVector3d(1.0,0.7,0.3));
			set_tet_vw(viewer_t1_removed,  global_faces_t1_removed,  Eigen::RowVector3d(0.8,0.2,0.2));
			set_tet_vw(viewer_t2_removed,  global_faces_t2_removed,  Eigen::RowVector3d(0.8,0.2,0.2));
			set_tet_vw(viewer_t12_removed, global_faces_t12_removed, Eigen::RowVector3d(0.8,0.2,0.2));

			// All valid combined
			{
				std::vector<std::array<index_t,3>> all_vf=global_faces_t1_valid;
				all_vf.insert(all_vf.end(),global_faces_t2_valid.begin(),global_faces_t2_valid.end());
				all_vf.insert(all_vf.end(),global_faces_t12_valid.begin(),global_faces_t12_valid.end());
				int nf=(int)all_vf.size();
				if(nf>0){
					Eigen::MatrixXi F(nf,3);
					for(int i=0;i<nf;++i) F.row(i)<<(int)all_vf[i][0],(int)all_vf[i][1],(int)all_vf[i][2];
					viewer_all_valid.data().set_mesh(V3d_a3,F);
					viewer_all_valid.data().show_lines=true; viewer_all_valid.data().show_faces=true;
					Eigen::MatrixXd C(nf,3); int off=0;
					for(int i=0;i<(int)global_faces_t1_valid.size();++i)  C.row(off++)<<0.4,0.6,1.0;
					for(int i=0;i<(int)global_faces_t2_valid.size();++i)  C.row(off++)<<0.4,1.0,0.4;
					for(int i=0;i<(int)global_faces_t12_valid.size();++i) C.row(off++)<<1.0,0.7,0.3;
					viewer_all_valid.data().set_colors(C);
				}else{viewer_all_valid.data().set_mesh(Vbox,Fbox); viewer_all_valid.data().show_faces=false;}
				set_tet_vw(viewer_all_valid_gray,all_vf,Eigen::RowVector3d(0.7,0.7,0.7));
				set_tet_vw(viewer_interactive,all_vf,Eigen::RowVector3d(0.7,0.7,0.7));
				global_interactive_faces=all_vf;
				global_interactive_face_types.assign(all_vf.size(),12);
			}

			// Free boundary via igl::boundary_facets on the tet matrix (same as algo 4)
			{
				// Count valid tets
				int n_valid_t1 = 0, n_valid_t2 = 0, n_valid_t12 = 0;
				for (const auto& info : global_t1_tets)
					if (info.centroid_in_solid && !(global_filter_t1_midpoint && info.fails_midpoint)) n_valid_t1++;
				for (const auto& info : global_t2_tets)
					if (info.centroid_in_solid && !(global_filter_t2_midpoint && info.fails_midpoint)) n_valid_t2++;
				for (const auto& info : global_t12_tets) {
					if (global_filter_t12_midpoint && info.fails_midpoint) continue;
					if (global_filter_t12_isolated && info.fails_isolated) continue;
					n_valid_t12++;
				}

				// Build T_all from per-tet vertex arrays.
				// In algo 3, T1/T2 orig_verts are LOCAL indices (no voffset) while
				// T12 verts already carry global indices (with voffset).
				// Derive global T1/T2 vertex indices from their face triples instead:
				// each tet has 4 triangular faces; the union of unique vertices across
				// all 4 faces gives the 4 tet vertices with global (voffset) indices.
				Eigen::MatrixXi T_all(n_valid_t1 + n_valid_t2 + n_valid_t12, 4);
				int ti = 0;
				for (const auto& info : global_t1_tets) {
					if (!info.centroid_in_solid) continue;
					if (global_filter_t1_midpoint && info.fails_midpoint) continue;
					// Collect 4 unique global vertex indices from the 4 triangular faces
					std::array<index_t,4> gv; int gc=0;
					for (int lf=0;lf<4&&gc<4;++lf)
						for (int k=0;k<3&&gc<4;++k) {
							index_t vid=info.faces[lf][k];
							bool found=false; for(int x=0;x<gc;++x) if(gv[x]==vid){found=true;break;}
							if(!found) gv[gc++]=vid;
						}
					for (int lv = 0; lv < 4; ++lv) T_all(ti, lv) = static_cast<int>(gv[lv]);
					ti++;
				}
				for (const auto& info : global_t2_tets) {
					if (!info.centroid_in_solid) continue;
					if (global_filter_t2_midpoint && info.fails_midpoint) continue;
					std::array<index_t,4> gv; int gc=0;
					for (int lf=0;lf<4&&gc<4;++lf)
						for (int k=0;k<3&&gc<4;++k) {
							index_t vid=info.faces[lf][k];
							bool found=false; for(int x=0;x<gc;++x) if(gv[x]==vid){found=true;break;}
							if(!found) gv[gc++]=vid;
						}
					for (int lv = 0; lv < 4; ++lv) T_all(ti, lv) = static_cast<int>(gv[lv]);
					ti++;
				}
				for (size_t i = 0; i < global_t12_tets.size(); ++i) {
					const auto& info = global_t12_tets[i];
					if (global_filter_t12_midpoint && info.fails_midpoint) continue;
					if (global_filter_t12_isolated && info.fails_isolated) continue;
					for (int lv = 0; lv < 4; ++lv) T_all(ti, lv) = static_cast<int>(info.verts[lv]);
					ti++;
				}
				T_all.conservativeResize(ti, 4);

				// Orient tets positively (same as algo 4)
				int reoriented = 0;
				for (int r = 0; r < ti; ++r) {
					Eigen::Vector3d p0 = V3d_a3.row(T_all(r,0));
					Eigen::Vector3d p1 = V3d_a3.row(T_all(r,1));
					Eigen::Vector3d p2 = V3d_a3.row(T_all(r,2));
					Eigen::Vector3d p3 = V3d_a3.row(T_all(r,3));
					double det = (p1-p0).dot((p2-p0).cross(p3-p0));
					if (det < 0) { std::swap(T_all(r,0), T_all(r,1)); reoriented++; }
				}
				std::cout << "  Tets for boundary_facets: " << ti
					<< " (T1v=" << n_valid_t1 << " T2v=" << n_valid_t2 << " T12v=" << n_valid_t12
					<< ", reoriented=" << reoriented << ")\n";

				Eigen::MatrixXi F_fb;
				igl::boundary_facets(T_all, F_fb);

				global_faces_free_boundary.clear();
				for (int i = 0; i < F_fb.rows(); ++i)
					global_faces_free_boundary.push_back({static_cast<index_t>(F_fb(i,0)), static_cast<index_t>(F_fb(i,1)), static_cast<index_t>(F_fb(i,2))});
				delaunay3d_free_boundary_faces = static_cast<int>(F_fb.rows());
				std::cout << "  Free boundary: " << delaunay3d_free_boundary_faces << " faces\n";

				// Reindex vertices for manifold check and OBJ export
				std::map<int, int> old_to_new;
				int next_id = 0;
				Eigen::MatrixXi F_fb_r(delaunay3d_free_boundary_faces, 3);
				for (int i = 0; i < delaunay3d_free_boundary_faces; ++i)
					for (int lv = 0; lv < 3; ++lv) {
						int ov = F_fb(i, lv);
						if (old_to_new.find(ov) == old_to_new.end()) old_to_new[ov] = next_id++;
						F_fb_r(i, lv) = old_to_new[ov];
					}
				Eigen::MatrixXd V_fb(next_id, 3);
				for (const auto& kv : old_to_new) V_fb.row(kv.second) = V3d_a3.row(kv.first);

				// Orient faces outward
				{
					Eigen::Vector3d centroid = V_fb.colwise().mean(); int flipped = 0;
					for (int i = 0; i < F_fb_r.rows(); ++i) {
						Eigen::Vector3d a=V_fb.row(F_fb_r(i,0)), b=V_fb.row(F_fb_r(i,1)), c=V_fb.row(F_fb_r(i,2));
						Eigen::Vector3d n=(b-a).cross(c-a);
						if (n.dot((a+b+c)/3.0 - centroid) < 0) { std::swap(F_fb_r(i,0), F_fb_r(i,1)); flipped++; }
					}
					std::cout << "  Faces oriented outward: flipped " << flipped << " / " << F_fb_r.rows() << "\n";
				}

				// Update global_faces_free_boundary to match reoriented faces
				{
					std::map<int,int> new_to_old; for (const auto& kv : old_to_new) new_to_old[kv.second] = kv.first;
					for (int i = 0; i < F_fb_r.rows(); ++i)
						global_faces_free_boundary[i] = {static_cast<index_t>(new_to_old[F_fb_r(i,0)]), static_cast<index_t>(new_to_old[F_fb_r(i,1)]), static_cast<index_t>(new_to_old[F_fb_r(i,2)])};
				}

				global_manifold_info = check_manifold(F_fb_r);
				std::cout << "\n  === MANIFOLD DIAGNOSTIC ===\n";
				std::cout << "  Edge-manifold:   " << (global_manifold_info.edge_manifold ? "YES" : "NO") << "\n";
				std::cout << "  Vertex-manifold: " << (global_manifold_info.vertex_manifold ? "YES" : "NO") << "\n";
				std::cout << "  Euler characteristic: " << global_manifold_info.euler_characteristic << "\n";
				if (!global_manifold_info.vertex_manifold)
					std::cout << "  Non-manifold vertices: " << global_manifold_info.non_manifold_vertices << "\n";
				if (!global_manifold_info.edge_manifold)
					std::cout << "  Non-manifold edges: " << global_manifold_info.non_manifold_edges << "\n";
				std::cout << "  Boundary vertices: " << next_id << "\n";
				std::cout << "  Result: " << (global_manifold_info.is_manifold() ? "MANIFOLD" : "NOT MANIFOLD") << "\n";
				std::cout << "  =========================\n";

				set_tet_vw(viewer_free_boundary, global_faces_free_boundary, Eigen::RowVector3d(0.7, 0.7, 0.7));
				std::string obj_path = "C:/Users/DELL/Desktop/Data/free_boundary.obj";
				if (igl::writeOBJ(obj_path, V_fb, F_fb_r))
					std::cout << "  OBJ: " << obj_path << "\n";
			}
		} // end algo 3 (disabled)

		{

		// =========================
		// ALGO 1/2/3/4: SINGLE DELAUNAY + FILTERS
		// (algo 3 uses same filters as algo 4 with 2D shape similarity groups for affinity)
		// =========================
		struct SimpleGroup4 {
			std::vector<int> l1_contour_ids;
			std::vector<int> l2_contour_ids;
			std::vector<solidpolygon> polys_l1, polys_l2;
			std::vector<polygon_reference> refs_l1, refs_l2;
			SolidRegionGrid grid_l1, grid_l2;
		};
		std::vector<SimpleGroup4> algo4_groups;

		if (recon_algo == 3 && !global_final_groups_2d.empty()) {
			// Build affinity groups from 2D shape similarity groups (global_final_groups_2d)
			std::cout << "\n=== ALGO3 2D Similarity Affinity Groups: " << global_final_groups_2d.size() << " ===\n";
			algo4_groups.reserve(global_final_groups_2d.size());
			for (const auto& fg : global_final_groups_2d) {
				if (fg.l1_side.empty() || fg.l2_side.empty()) continue;
				SimpleGroup4 grp;
				for (const auto& pp : fg.l1_side) {
					solidpolygon P;
					P.contours = pp.contour_ids;
					grp.refs_l1.push_back(polygon_reference((int)grp.polys_l1.size(), 0));
					grp.polys_l1.push_back(P);
				}
				for (const auto& pp : fg.l2_side) {
					solidpolygon P;
					P.contours = pp.contour_ids;
					grp.refs_l2.push_back(polygon_reference((int)grp.polys_l2.size(), 1));
					grp.polys_l2.push_back(P);
				}
				algo4_groups.push_back(std::move(grp));
			}
			// Fix dangling pointers after push_back(move)
			for (auto& grp : algo4_groups) {
				grp.grid_l1.stored_poly_refs = &grp.refs_l1;
				grp.grid_l1.stored_all_polys = &grp.polys_l1;
				grp.grid_l1.stored_contours  = &levels[startIdx-1].contours;
				grp.grid_l1.valid = true;
				grp.grid_l2.stored_poly_refs = &grp.refs_l2;
				grp.grid_l2.stored_all_polys = &grp.polys_l2;
				grp.grid_l2.stored_contours  = &levels[startIdx].contours;
				grp.grid_l2.valid = true;
			}
			std::cout << "  Built " << algo4_groups.size() << " affinity groups from 2D shape similarity\n";
		} else if (recon_algo == 4) {
			// Use the solid-solid mapping groups already computed from the forest
			// (same as global_mapping_groups - no depuration, no hole-hole)
			std::cout << "\n=== ALGO4 Solid-Solid Mapping Groups: " << global_mapping_groups.size() << " ===\n";
			for (int gi = 0; gi < (int)global_mapping_groups.size(); ++gi) {
				const auto& mg = global_mapping_groups[gi];
				std::cout << "  Group " << (gi+1) << ": L1={";
				for (int i : mg.polygons_l1) std::cout << format_polygon_name_l1(&polygon_refs_l1[i], all_polygons, forest_l1);
				std::cout << "} vs L2={";
				for (int i : mg.polygons_l2) std::cout << format_polygon_name_l2(&polygon_refs_l2[i], all_polygons, forest_l2);
				std::cout << "}";
				if (mg.polygons_l1.empty() || mg.polygons_l2.empty()) std::cout << " [EMPTY SIDE - skipped]";
				std::cout << "\n";
			}

			algo4_groups.reserve(global_mapping_groups.size());
			for (const auto& mg : global_mapping_groups) {
				if (mg.polygons_l1.empty() || mg.polygons_l2.empty()) continue;
				SimpleGroup4 grp;
				for (int pid : mg.polygons_l1) {
					grp.l1_contour_ids.push_back(pid);
					grp.refs_l1.push_back(polygon_reference((int)grp.polys_l1.size(), 0));
					grp.polys_l1.push_back(all_polygons[polygon_refs_l1[pid].polygon_id]);
				}
				for (int pid : mg.polygons_l2) {
					grp.l2_contour_ids.push_back(pid);
					grp.refs_l2.push_back(polygon_reference((int)grp.polys_l2.size(), 1));
					grp.polys_l2.push_back(all_polygons[polygon_refs_l2[pid].polygon_id]);
				}
				algo4_groups.push_back(std::move(grp));
			}
			// Fix stored pointers after all push_backs (vector may have reallocated)
			for (auto& grp : algo4_groups) {
				grp.grid_l1.stored_poly_refs = &grp.refs_l1;
				grp.grid_l1.stored_all_polys = &grp.polys_l1;
				grp.grid_l1.stored_contours  = &levels[startIdx-1].contours;
				grp.grid_l1.valid = true;
				grp.grid_l2.stored_poly_refs = &grp.refs_l2;
				grp.grid_l2.stored_all_polys = &grp.polys_l2;
				grp.grid_l2.stored_contours  = &levels[startIdx].contours;
				grp.grid_l2.valid = true;
			}
			std::cout << "  Grids built for " << algo4_groups.size() << " groups\n";
		}

		// Use resampled contours for Voronoi
		const auto& vor_contours_l1 = (resampleFactor > 1) ? resampled_l1 : levels[startIdx - 1].contours;
		const auto& vor_contours_l2 = (resampleFactor > 1) ? resampled_l2 : levels[startIdx].contours;

		// Precompute solid region grids for O(1) point classification
		SolidRegionGrid sp_vor_grid_l1, sp_vor_grid_l2;
		sp_vor_grid_l1.precompute(polygon_refs_l1, all_polygons, levels[startIdx - 1].contours);
		sp_vor_grid_l2.precompute(polygon_refs_l2, all_polygons, levels[startIdx].contours);
		std::cout << "\nSolid region grids: L1=" << sp_vor_grid_l1.nx << "x" << sp_vor_grid_l1.ny
			<< ", L2=" << sp_vor_grid_l2.nx << "x" << sp_vor_grid_l2.ny << "\n";

		// =========================
		// VORONOI L1 - DATA + CLASSIFY
		// =========================
		auto vor_edges_l1 = compute_voronoi_edges(vor_contours_l1);
		medial_axes_result ma_l1;
		{

			for (const auto& contour : vor_contours_l1) {
				int n = static_cast<int>(contour.points.size());
				if (n == 0) continue;
				Eigen::MatrixXd V(n, 3);
				for (int j = 0; j < n; ++j)
					V.row(j) = contour.points[j];
				viewer_vor_l1.data().add_points(V, Eigen::RowVector3d(1, 0, 0));
				for (int j = 0; j < n; ++j)
					viewer_vor_l1.data().add_edges(
						V.row(j), V.row((j + 1) % n),
						Eigen::RowVector3d(0, 0, 0));
			}

			for (const auto& e : vor_edges_l1)
				viewer_vor_l1.data().add_edges(
					Eigen::RowVector3d(e.first.x, e.first.y, z_l1),
					Eigen::RowVector3d(e.second.x, e.second.y, z_l1),
					Eigen::RowVector3d(0, 0, 1));

			viewer_vor_l1.data().set_mesh(Vbox, Fbox);
			viewer_vor_l1.data().show_faces = false;
			viewer_vor_l1.data().point_size = 4.0;

			std::cout << "\nVoronoi L1: " << vor_edges_l1.size() << " edges\n";

			// Classify into IMA / EMA
			ma_l1 = classify_voronoi_edges(
				vor_edges_l1, polygon_refs_l1, all_polygons,
				levels[startIdx - 1].contours, &sp_vor_grid_l1);

			ima_l1_count = static_cast<int>(ma_l1.ima_edges.size());
			ema_l1_count = static_cast<int>(ma_l1.ema_edges.size());
			disc_l1_count = static_cast<int>(ma_l1.discarded_edges.size());

			std::cout << "  IMA L1: " << ima_l1_count << " edges\n";
			std::cout << "  EMA L1: " << ema_l1_count << " edges\n";
			std::cout << "  Discarded L1: " << disc_l1_count << " edges\n";

			// IMA L1 viewer data
			for (const auto& contour : levels[startIdx - 1].contours) {
				int n = static_cast<int>(contour.points.size());
				if (n == 0) continue;
				Eigen::MatrixXd V(n, 3);
				for (int j = 0; j < n; ++j)
					V.row(j) = contour.points[j];
				for (int j = 0; j < n; ++j)
					viewer_ima_l1.data().add_edges(
						V.row(j), V.row((j + 1) % n),
						Eigen::RowVector3d(0, 0, 0));
			}
			for (const auto& e : ma_l1.ima_edges)
				viewer_ima_l1.data().add_edges(
					Eigen::RowVector3d(e.first.x, e.first.y, z_l1),
					Eigen::RowVector3d(e.second.x, e.second.y, z_l1),
					Eigen::RowVector3d(0, 0, 1));
			viewer_ima_l1.data().set_mesh(Vbox, Fbox);
			viewer_ima_l1.data().show_faces = false;
			viewer_ima_l1.data().point_size = 2.0;

			// EMA L1 viewer data
			for (const auto& contour : levels[startIdx - 1].contours) {
				int n = static_cast<int>(contour.points.size());
				if (n == 0) continue;
				Eigen::MatrixXd V(n, 3);
				for (int j = 0; j < n; ++j)
					V.row(j) = contour.points[j];
				for (int j = 0; j < n; ++j)
					viewer_ema_l1.data().add_edges(
						V.row(j), V.row((j + 1) % n),
						Eigen::RowVector3d(0, 0, 0));
			}
			for (const auto& e : ma_l1.ema_edges)
				viewer_ema_l1.data().add_edges(
					Eigen::RowVector3d(e.first.x, e.first.y, z_l1),
					Eigen::RowVector3d(e.second.x, e.second.y, z_l1),
					Eigen::RowVector3d(0.8, 0, 0));
			viewer_ema_l1.data().set_mesh(Vbox, Fbox);
			viewer_ema_l1.data().show_faces = false;
			viewer_ema_l1.data().point_size = 2.0;

			// IMA+EMA L1 combined viewer data
			for (const auto& contour : levels[startIdx - 1].contours) {
				int n = static_cast<int>(contour.points.size());
				if (n == 0) continue;
				Eigen::MatrixXd V(n, 3);
				for (int j = 0; j < n; ++j)
					V.row(j) = contour.points[j];
				for (int j = 0; j < n; ++j)
					viewer_both_l1.data().add_edges(
						V.row(j), V.row((j + 1) % n),
						Eigen::RowVector3d(0, 0, 0));
			}
			for (const auto& e : ma_l1.ima_edges)
				viewer_both_l1.data().add_edges(
					Eigen::RowVector3d(e.first.x, e.first.y, z_l1),
					Eigen::RowVector3d(e.second.x, e.second.y, z_l1),
					Eigen::RowVector3d(0, 0, 1));
			for (const auto& e : ma_l1.ema_edges)
				viewer_both_l1.data().add_edges(
					Eigen::RowVector3d(e.first.x, e.first.y, z_l1),
					Eigen::RowVector3d(e.second.x, e.second.y, z_l1),
					Eigen::RowVector3d(0.8, 0, 0));
			viewer_both_l1.data().set_mesh(Vbox, Fbox);
			viewer_both_l1.data().show_faces = false;
			viewer_both_l1.data().point_size = 2.0;
		}

		// =========================
		// VORONOI L2 - DATA + CLASSIFY
		// =========================
		auto vor_edges_l2 = compute_voronoi_edges(vor_contours_l2);
		medial_axes_result ma_l2;
		{

			for (const auto& contour : vor_contours_l2) {
				int n = static_cast<int>(contour.points.size());
				if (n == 0) continue;
				Eigen::MatrixXd V(n, 3);
				for (int j = 0; j < n; ++j)
					V.row(j) = contour.points[j];
				viewer_vor_l2.data().add_points(V, Eigen::RowVector3d(1, 0, 0));
				for (int j = 0; j < n; ++j)
					viewer_vor_l2.data().add_edges(
						V.row(j), V.row((j + 1) % n),
						Eigen::RowVector3d(0, 0, 0));
			}

			for (const auto& e : vor_edges_l2)
				viewer_vor_l2.data().add_edges(
					Eigen::RowVector3d(e.first.x, e.first.y, z_l2),
					Eigen::RowVector3d(e.second.x, e.second.y, z_l2),
					Eigen::RowVector3d(0, 0, 1));

			viewer_vor_l2.data().set_mesh(Vbox, Fbox);
			viewer_vor_l2.data().show_faces = false;
			viewer_vor_l2.data().point_size = 4.0;

			std::cout << "\nVoronoi L2: " << vor_edges_l2.size() << " edges\n";

			// Classify into IMA / EMA
			ma_l2 = classify_voronoi_edges(
				vor_edges_l2, polygon_refs_l2, all_polygons,
				levels[startIdx].contours, &sp_vor_grid_l2);

			ima_l2_count = static_cast<int>(ma_l2.ima_edges.size());
			ema_l2_count = static_cast<int>(ma_l2.ema_edges.size());
			disc_l2_count = static_cast<int>(ma_l2.discarded_edges.size());

			std::cout << "  IMA L2: " << ima_l2_count << " edges\n";
			std::cout << "  EMA L2: " << ema_l2_count << " edges\n";
			std::cout << "  Discarded L2: " << disc_l2_count << " edges\n";

			// IMA L2 viewer data
			for (const auto& contour : levels[startIdx].contours) {
				int n = static_cast<int>(contour.points.size());
				if (n == 0) continue;
				Eigen::MatrixXd V(n, 3);
				for (int j = 0; j < n; ++j)
					V.row(j) = contour.points[j];
				for (int j = 0; j < n; ++j)
					viewer_ima_l2.data().add_edges(
						V.row(j), V.row((j + 1) % n),
						Eigen::RowVector3d(0, 0, 0));
			}
			for (const auto& e : ma_l2.ima_edges)
				viewer_ima_l2.data().add_edges(
					Eigen::RowVector3d(e.first.x, e.first.y, z_l2),
					Eigen::RowVector3d(e.second.x, e.second.y, z_l2),
					Eigen::RowVector3d(0, 0, 1));
			viewer_ima_l2.data().set_mesh(Vbox, Fbox);
			viewer_ima_l2.data().show_faces = false;
			viewer_ima_l2.data().point_size = 2.0;

			// EMA L2 viewer data
			for (const auto& contour : levels[startIdx].contours) {
				int n = static_cast<int>(contour.points.size());
				if (n == 0) continue;
				Eigen::MatrixXd V(n, 3);
				for (int j = 0; j < n; ++j)
					V.row(j) = contour.points[j];
				for (int j = 0; j < n; ++j)
					viewer_ema_l2.data().add_edges(
						V.row(j), V.row((j + 1) % n),
						Eigen::RowVector3d(0, 0, 0));
			}
			for (const auto& e : ma_l2.ema_edges)
				viewer_ema_l2.data().add_edges(
					Eigen::RowVector3d(e.first.x, e.first.y, z_l2),
					Eigen::RowVector3d(e.second.x, e.second.y, z_l2),
					Eigen::RowVector3d(0.8, 0, 0));
			viewer_ema_l2.data().set_mesh(Vbox, Fbox);
			viewer_ema_l2.data().show_faces = false;
			viewer_ema_l2.data().point_size = 2.0;

			// IMA+EMA L2 combined viewer data
			for (const auto& contour : levels[startIdx].contours) {
				int n = static_cast<int>(contour.points.size());
				if (n == 0) continue;
				Eigen::MatrixXd V(n, 3);
				for (int j = 0; j < n; ++j)
					V.row(j) = contour.points[j];
				for (int j = 0; j < n; ++j)
					viewer_both_l2.data().add_edges(
						V.row(j), V.row((j + 1) % n),
						Eigen::RowVector3d(0, 0, 0));
			}
			for (const auto& e : ma_l2.ima_edges)
				viewer_both_l2.data().add_edges(
					Eigen::RowVector3d(e.first.x, e.first.y, z_l2),
					Eigen::RowVector3d(e.second.x, e.second.y, z_l2),
					Eigen::RowVector3d(0, 0, 1));
			for (const auto& e : ma_l2.ema_edges)
				viewer_both_l2.data().add_edges(
					Eigen::RowVector3d(e.first.x, e.first.y, z_l2),
					Eigen::RowVector3d(e.second.x, e.second.y, z_l2),
					Eigen::RowVector3d(0.8, 0, 0));
			viewer_both_l2.data().set_mesh(Vbox, Fbox);
			viewer_both_l2.data().show_faces = false;
			viewer_both_l2.data().point_size = 2.0;
		}

		// =========================
		// EMA PROJECTION & AUGMENTED POINTS
		// =========================

		// EMA L1 projected onto L2
		ema_projection_result proj_ema_l1_on_l2 = compute_ema_projection(
			ma_l1.ema_edges, polygon_refs_l2, all_polygons,
			levels[startIdx].contours, &sp_vor_grid_l2);

		// EMA L2 projected onto L1
		ema_projection_result proj_ema_l2_on_l1 = compute_ema_projection(
			ma_l2.ema_edges, polygon_refs_l1, all_polygons,
			levels[startIdx - 1].contours, &sp_vor_grid_l1);

		std::cout << "\n=================================\n";
		std::cout << "EMA PROJECTION\n";
		std::cout << "=================================\n";
		std::cout << "EMA L1 -> L2: " << proj_ema_l1_on_l2.solid_vertices.size()
			<< " solid, " << proj_ema_l1_on_l2.void_vertices.size() << " void\n";
		std::cout << "EMA L2 -> L1: " << proj_ema_l2_on_l1.solid_vertices.size()
			<< " solid, " << proj_ema_l2_on_l1.void_vertices.size() << " void\n";

		// Build augmented points L1
		augmented_points_l1.clear();
		{
			const auto& src = (resampleFactor > 1) ? resampled_l1 : levels[startIdx - 1].contours;
			for (const auto& c : src)
				for (const auto& p : c.points)
					augmented_points_l1.push_back(p);
			aug_contour_l1 = static_cast<int>(augmented_points_l1.size());

			if (recon_algo >= 2) {
				for (const auto& e : ma_l1.ima_edges) {
					augmented_points_l1.emplace_back(e.first.x, e.first.y, z_l1);
					augmented_points_l1.emplace_back(e.second.x, e.second.y, z_l1);
				}
			}
			aug_ima_l1 = static_cast<int>(augmented_points_l1.size()) - aug_contour_l1;

			for (const auto& v : proj_ema_l2_on_l1.solid_vertices)
				augmented_points_l1.emplace_back(v.x(), v.y(), z_l1);
			aug_ema_proj_l1 = static_cast<int>(proj_ema_l2_on_l1.solid_vertices.size());
		}

		// Build augmented points L2
		augmented_points_l2.clear();
		{
			const auto& src = (resampleFactor > 1) ? resampled_l2 : levels[startIdx].contours;
			for (const auto& c : src)
				for (const auto& p : c.points)
					augmented_points_l2.push_back(p);
			aug_contour_l2 = static_cast<int>(augmented_points_l2.size());

			if (recon_algo >= 2) {
				for (const auto& e : ma_l2.ima_edges) {
					augmented_points_l2.emplace_back(e.first.x, e.first.y, z_l2);
					augmented_points_l2.emplace_back(e.second.x, e.second.y, z_l2);
				}
			}
			aug_ima_l2 = static_cast<int>(augmented_points_l2.size()) - aug_contour_l2;

			for (const auto& v : proj_ema_l1_on_l2.solid_vertices)
				augmented_points_l2.emplace_back(v.x(), v.y(), z_l2);
			aug_ema_proj_l2 = static_cast<int>(proj_ema_l1_on_l2.solid_vertices.size());
		}

		// Set global IMA vertex index ranges for the isolated rescue filter
		global_ima_start_l1 = static_cast<index_t>(aug_contour_l1);
		global_ima_end_l1 = static_cast<index_t>(aug_contour_l1 + aug_ima_l1);
		{
			index_t n_l1_pts = static_cast<index_t>(augmented_points_l1.size());
			global_ima_start_l2 = n_l1_pts + static_cast<index_t>(aug_contour_l2);
			global_ima_end_l2 = n_l1_pts + static_cast<index_t>(aug_contour_l2 + aug_ima_l2);
		}

		std::cout << "\n=================================\n";
		std::cout << "AUGMENTED POINTS\n";
		std::cout << "=================================\n";
		std::cout << "L1: " << augmented_points_l1.size() << " total"
			<< " (contour=" << aug_contour_l1
			<< ", IMA=" << aug_ima_l1
			<< ", EMA_proj=" << aug_ema_proj_l1 << ")\n";
		std::cout << "L2: " << augmented_points_l2.size() << " total"
			<< " (contour=" << aug_contour_l2
			<< ", IMA=" << aug_ima_l2
			<< ", EMA_proj=" << aug_ema_proj_l2 << ")\n";
		std::cout << "IMA ranges: L1=[" << global_ima_start_l1 << ".." << global_ima_end_l1
			<< "), L2=[" << global_ima_start_l2 << ".." << global_ima_end_l2 << ")\n";
		std::cout << "=================================\n";

		// =========================
		// 3D DELAUNAY TRIANGULATION
		// =========================
		{
			std::vector<double> coords_3d;
			for (const auto& p : augmented_points_l1) {
				coords_3d.push_back(p.x());
				coords_3d.push_back(p.y());
				coords_3d.push_back(p.z());
			}
			for (const auto& p : augmented_points_l2) {
				coords_3d.push_back(p.x());
				coords_3d.push_back(p.y());
				coords_3d.push_back(p.z());
			}

			index_t N3d = coords_3d.size() / 3;
			delaunay3d_num_points = static_cast<int>(N3d);

			std::cout << "\n=================================\n";
			std::cout << "3D DELAUNAY TRIANGULATION\n";
			std::cout << "=================================\n";
			std::cout << "Input points: " << N3d
				<< " (L1=" << augmented_points_l1.size()
				<< ", L2=" << augmented_points_l2.size() << ")\n";

			if (N3d >= 4) {
				Delaunay_var delaunay3d = Delaunay::create(3, "BDEL");
				delaunay3d->set_vertices(N3d, coords_3d.data());

				index_t nb_tets = delaunay3d->nb_cells();
				delaunay3d_num_tets = static_cast<int>(nb_tets);

				std::cout << "Tetrahedra: " << nb_tets << "\n";

				// Extract boundary faces: a face is boundary if its adjacent tet is -1
				std::vector<std::array<index_t, 3>> boundary_faces;

				// For each tet, 4 faces. Face i is opposite to vertex i.
				// Face opposite vertex 0: vertices 1,2,3
				// Face opposite vertex 1: vertices 0,2,3
				// Face opposite vertex 2: vertices 0,1,3
				// Face opposite vertex 3: vertices 0,1,2
				const int face_vertex[4][3] = {
					{1, 2, 3},
					{0, 2, 3},
					{0, 1, 3},
					{0, 1, 2}
				};

				for (index_t t = 0; t < nb_tets; ++t) {
					for (index_t lf = 0; lf < 4; ++lf) {
						signed_index_t adj = delaunay3d->cell_adjacent(t, lf);
						if (adj == NO_INDEX) {
							const signed_index_t* v = delaunay3d->cell_to_v() + 4 * t;
							boundary_faces.push_back({
								static_cast<index_t>(v[face_vertex[lf][0]]),
								static_cast<index_t>(v[face_vertex[lf][1]]),
								static_cast<index_t>(v[face_vertex[lf][2]])
							});
						}
					}
				}

				delaunay3d_num_faces = static_cast<int>(boundary_faces.size());
				std::cout << "Boundary faces: " << delaunay3d_num_faces << "\n";

				// Classify tetrahedra: T1 (3 in L1, 1 in L2), T2 (3 in L2, 1 in L1), T12 (2+2)
				index_t n_l1_pts = static_cast<index_t>(augmented_points_l1.size());
				delaunay3d_T1 = 0;
				delaunay3d_T2 = 0;
				delaunay3d_T12 = 0;
				delaunay3d_degenerate = 0;

				for (index_t t = 0; t < nb_tets; ++t) {
					const signed_index_t* v = delaunay3d->cell_to_v() + 4 * t;
					int count_l1 = 0;
					for (int lv = 0; lv < 4; ++lv) {
						if (static_cast<index_t>(v[lv]) < n_l1_pts)
							count_l1++;
					}
					switch (count_l1) {
						case 3: delaunay3d_T1++; break;
						case 1: delaunay3d_T2++; break;
						case 2: delaunay3d_T12++; break;
						default: delaunay3d_degenerate++; break;
					}
				}

				std::cout << "Classification:\n";
				std::cout << "  T1  (3 L1 + 1 L2): " << delaunay3d_T1 << "\n";
				std::cout << "  T2  (1 L1 + 3 L2): " << delaunay3d_T2 << "\n";
				std::cout << "  T12 (2 L1 + 2 L2): " << delaunay3d_T12 << "\n";
				if (delaunay3d_degenerate > 0)
					std::cout << "  Degenerate (4+0):   " << delaunay3d_degenerate << "\n";

				// Extract all 4 faces per tetrahedron, classified by type
				std::vector<std::array<index_t, 3>> faces_t1, faces_t2, faces_t12;

				for (index_t t = 0; t < nb_tets; ++t) {
					const signed_index_t* v = delaunay3d->cell_to_v() + 4 * t;
					int count_l1 = 0;
					for (int lv = 0; lv < 4; ++lv) {
						if (static_cast<index_t>(v[lv]) < n_l1_pts)
							count_l1++;
					}

					std::vector<std::array<index_t, 3>>* target = nullptr;
					if (count_l1 == 3) target = &faces_t1;
					else if (count_l1 == 1) target = &faces_t2;
					else if (count_l1 == 2) target = &faces_t12;

					if (target) {
						for (int lf = 0; lf < 4; ++lf) {
							target->push_back({
								static_cast<index_t>(v[face_vertex[lf][0]]),
								static_cast<index_t>(v[face_vertex[lf][1]]),
								static_cast<index_t>(v[face_vertex[lf][2]])
							});
						}
					}
				}

			delaunay3d_T1_faces = static_cast<int>(faces_t1.size());
				delaunay3d_T2_faces = static_cast<int>(faces_t2.size());
				delaunay3d_T12_faces = static_cast<int>(faces_t12.size());

				std::cout << "Faces per type:\n";
				std::cout << "  T1:  " << delaunay3d_T1_faces << "\n";
				std::cout << "  T2:  " << delaunay3d_T2_faces << "\n";
				std::cout << "  T12: " << delaunay3d_T12_faces << "\n";

				// =========================
				// FILTER T1 and T2 tetrahedra
				// =========================
				const auto& filter_contours_l1 = levels[startIdx - 1].contours;
				const auto& filter_contours_l2 = levels[startIdx].contours;

				// Reuse grids from outer scope (sp_vor_grid_l1/l2)
				const auto& sp_grid_l1 = sp_vor_grid_l1;
				const auto& sp_grid_l2 = sp_vor_grid_l2;

				// Store globally for IMA rescue segment validation
				global_n_l1_pts = n_l1_pts;
				global_filter_contours_l1 = &filter_contours_l1;
				global_filter_contours_l2 = &filter_contours_l2;
				global_grid_l1 = &sp_vor_grid_l1;
				global_grid_l2 = &sp_vor_grid_l2;

				std::vector<std::array<index_t, 3>> faces_t1_valid, faces_t1_removed;
				std::vector<std::array<index_t, 3>> faces_t2_valid, faces_t2_removed;

				delaunay3d_T1_valid = 0;
				delaunay3d_T1_removed = 0;
				delaunay3d_T2_valid = 0;
				delaunay3d_T2_removed = 0;

				global_t1_tets.clear();
				global_t2_tets.clear();

				for (index_t t = 0; t < nb_tets; ++t) {
					const signed_index_t* v = delaunay3d->cell_to_v() + 4 * t;
					int count_l1 = 0;
					index_t base_verts[3];
					int base_count = 0;
					for (int lv = 0; lv < 4; ++lv) {
						if (static_cast<index_t>(v[lv]) < n_l1_pts)
							count_l1++;
					}

					if (count_l1 == 3) {
						// T1: base is the 3 vertices in L1
						t1_t2_tet_info info;
						info.type = 1;
						base_count = 0;
						info.apex_vert = 0;
						for (int lv = 0; lv < 4; ++lv) {
							info.orig_verts[lv] = static_cast<index_t>(v[lv]);
							if (static_cast<index_t>(v[lv]) < n_l1_pts)
								info.base_verts[base_count++] = static_cast<index_t>(v[lv]);
							else
								info.apex_vert = static_cast<index_t>(v[lv]);
						}
						for (int lf = 0; lf < 4; ++lf) {
							info.faces[lf] = {
								static_cast<index_t>(v[face_vertex[lf][0]]),
								static_cast<index_t>(v[face_vertex[lf][1]]),
								static_cast<index_t>(v[face_vertex[lf][2]])
							};
						}

						info.centroid_x = (coords_3d[3*info.base_verts[0]] + coords_3d[3*info.base_verts[1]] + coords_3d[3*info.base_verts[2]]) / 3.0;
						info.centroid_y = (coords_3d[3*info.base_verts[0]+1] + coords_3d[3*info.base_verts[1]+1] + coords_3d[3*info.base_verts[2]+1]) / 3.0;
						Eigen::Vector2d centroid(info.centroid_x, info.centroid_y);

						info.centroid_in_solid = (sp_grid_l1.classify(info.centroid_x, info.centroid_y) >= 0);

						info.void_l1_count = 0;
						info.void_l2_count = 0;
						info.fails_midpoint = false;
						{
							const double sample_t[3] = { 2.0/5.0, 0.5, 3.0/5.0 };
							bool all_void_l1 = true, all_void_l2 = true;
							for (int be = 0; be < 3; ++be) {
								double bx = coords_3d[3*info.base_verts[be]];
								double by = coords_3d[3*info.base_verts[be]+1];
								double ax = coords_3d[3*info.apex_vert];
								double ay = coords_3d[3*info.apex_vert+1];
								for (int si = 0; si < 3; ++si) {
									double t = sample_t[si];
									double mx = bx + t * (ax - bx);
									double my = by + t * (ay - by);
									info.midpoint_x[be][si] = mx;
									info.midpoint_y[be][si] = my;
									info.midpoint_in_solid_l1[be][si] = (sp_grid_l1.classify(mx, my) >= 0);
									info.midpoint_in_solid_l2[be][si] = (sp_grid_l2.classify(mx, my) >= 0);
									if (!info.midpoint_in_solid_l1[be][si]) info.void_l1_count++;
									if (!info.midpoint_in_solid_l2[be][si]) info.void_l2_count++;
									if (info.midpoint_in_solid_l1[be][si]) all_void_l1 = false;
									if (info.midpoint_in_solid_l2[be][si]) all_void_l2 = false;
								}
							}
							if (all_void_l1 && all_void_l2)
								info.fails_midpoint = true;
						}

						global_t1_tets.push_back(info);
					}
					else if (count_l1 == 1) {
						// T2: base is the 3 vertices in L2
						t1_t2_tet_info info;
						info.type = 2;
						base_count = 0;
						info.apex_vert = 0;
						for (int lv = 0; lv < 4; ++lv) {
							info.orig_verts[lv] = static_cast<index_t>(v[lv]);
							if (static_cast<index_t>(v[lv]) >= n_l1_pts)
								info.base_verts[base_count++] = static_cast<index_t>(v[lv]);
							else
								info.apex_vert = static_cast<index_t>(v[lv]);
						}
						for (int lf = 0; lf < 4; ++lf) {
							info.faces[lf] = {
								static_cast<index_t>(v[face_vertex[lf][0]]),
								static_cast<index_t>(v[face_vertex[lf][1]]),
								static_cast<index_t>(v[face_vertex[lf][2]])
							};
						}

						info.centroid_x = (coords_3d[3*info.base_verts[0]] + coords_3d[3*info.base_verts[1]] + coords_3d[3*info.base_verts[2]]) / 3.0;
						info.centroid_y = (coords_3d[3*info.base_verts[0]+1] + coords_3d[3*info.base_verts[1]+1] + coords_3d[3*info.base_verts[2]+1]) / 3.0;
						Eigen::Vector2d centroid(info.centroid_x, info.centroid_y);

						info.centroid_in_solid = (sp_grid_l2.classify(info.centroid_x, info.centroid_y) >= 0);

						info.void_l1_count = 0;
						info.void_l2_count = 0;
						info.fails_midpoint = false;
						{
							const double sample_t[3] = { 2.0/5.0, 0.5, 3.0/5.0 };
							bool all_void_l1 = true, all_void_l2 = true;
							for (int be = 0; be < 3; ++be) {
								double bx = coords_3d[3*info.base_verts[be]];
								double by = coords_3d[3*info.base_verts[be]+1];
								double ax = coords_3d[3*info.apex_vert];
								double ay = coords_3d[3*info.apex_vert+1];
								for (int si = 0; si < 3; ++si) {
									double t = sample_t[si];
									double mx = bx + t * (ax - bx);
									double my = by + t * (ay - by);
									info.midpoint_x[be][si] = mx;
									info.midpoint_y[be][si] = my;
									info.midpoint_in_solid_l1[be][si] = (sp_grid_l1.classify(mx, my) >= 0);
									info.midpoint_in_solid_l2[be][si] = (sp_grid_l2.classify(mx, my) >= 0);
									if (!info.midpoint_in_solid_l1[be][si]) info.void_l1_count++;
									if (!info.midpoint_in_solid_l2[be][si]) info.void_l2_count++;
									if (info.midpoint_in_solid_l1[be][si]) all_void_l1 = false;
									if (info.midpoint_in_solid_l2[be][si]) all_void_l2 = false;
								}
							}
							if (all_void_l1 && all_void_l2)
								info.fails_midpoint = true;
						}

						global_t2_tets.push_back(info);
					}
				}

				// Compute T1/T2 valid/removed from per-tet flags
				recompute_t1_t2_from_filters();
				faces_t1_valid = global_faces_t1_valid;
				faces_t1_removed = global_faces_t1_removed;
				faces_t2_valid = global_faces_t2_valid;
				faces_t2_removed = global_faces_t2_removed;

				// =========================
				// FILTER T12 tetrahedra (midpoint + isolated + IMA rescue)
				// =========================
				std::vector<std::array<index_t, 3>> faces_t12_valid, faces_t12_removed;

				delaunay3d_T12_valid = 0;
				delaunay3d_T12_removed = 0;

				{
					std::cout << "\nT12 filter:\n";

					int t12_edge_not_in_solid = 0;
					int t12_isolated_count = 0;

					global_t12_tets.clear();

					for (index_t t = 0; t < nb_tets; ++t) {
						const signed_index_t* v = delaunay3d->cell_to_v() + 4 * t;
						int count_l1 = 0;
						index_t verts_l1_global[2], verts_l2_global[2];
						int cl1 = 0, cl2 = 0;
						for (int lv = 0; lv < 4; ++lv) {
							if (static_cast<index_t>(v[lv]) < n_l1_pts) {
								count_l1++;
								if (cl1 < 2) verts_l1_global[cl1++] = static_cast<index_t>(v[lv]);
							} else {
								if (cl2 < 2) verts_l2_global[cl2++] = static_cast<index_t>(v[lv]);
							}
						}

						if (count_l1 != 2) continue; // Not T12

						t12_tet_info info;
												info.fails_midpoint = false;
						info.fails_isolated = false;

						// Store 4 vertices
						for (int lv = 0; lv < 4; ++lv)
							info.verts[lv] = static_cast<index_t>(v[lv]);

						// Store 4 faces
						for (int lf = 0; lf < 4; ++lf) {
							info.faces[lf] = {
								static_cast<index_t>(v[face_vertex[lf][0]]),
								static_cast<index_t>(v[face_vertex[lf][1]]),
								static_cast<index_t>(v[face_vertex[lf][2]])
							};
						}

						// Midpoint filter: midpoint of each Delaunay edge must be in solid region
						// Points on the contour boundary are treated as inside (not void)
						{
							double mx_l1 = (coords_3d[3 * verts_l1_global[0]]     + coords_3d[3 * verts_l1_global[1]])     * 0.5;
							double my_l1 = (coords_3d[3 * verts_l1_global[0] + 1] + coords_3d[3 * verts_l1_global[1] + 1]) * 0.5;
							Eigen::Vector2d mid_l1(mx_l1, my_l1);
							int region_l1 = sp_grid_l1.classify(mx_l1, my_l1);
							if (region_l1 < 0)
								if (point_near_contour_boundary(mid_l1, polygon_refs_l1, all_polygons, filter_contours_l1))
									region_l1 = 0;

							double mx_l2 = (coords_3d[3 * verts_l2_global[0]]     + coords_3d[3 * verts_l2_global[1]])     * 0.5;
							double my_l2 = (coords_3d[3 * verts_l2_global[0] + 1] + coords_3d[3 * verts_l2_global[1] + 1]) * 0.5;
							Eigen::Vector2d mid_l2(mx_l2, my_l2);
							int region_l2 = sp_grid_l2.classify(mx_l2, my_l2);
							if (region_l2 < 0)
								if (point_near_contour_boundary(mid_l2, polygon_refs_l2, all_polygons, filter_contours_l2))
									region_l2 = 0;

							if (region_l1 < 0 || region_l2 < 0) {
								info.fails_midpoint = true;
								t12_edge_not_in_solid++;
							}
						}

						// Isolated filter is computed after all T12 tets are collected (direct-touch check below)

						global_t12_tets.push_back(info);
					}

					// =========================
					// Isolated filter: a T12 is valid only if it directly shares
					// a triangular face with a valid T1 or T2 tetrahedron.
					// =========================
					{
						// Helper: create a canonical face key (sorted triple)
						auto make_face_key = [](index_t a, index_t b, index_t c) -> std::array<index_t, 3> {
							std::array<index_t, 3> f = {a, b, c};
							if (f[0] > f[1]) std::swap(f[0], f[1]);
							if (f[1] > f[2]) std::swap(f[1], f[2]);
							if (f[0] > f[1]) std::swap(f[0], f[1]);
							return f;
						};

						// Step 1: Build face set from already-filtered valid T1 and T2
						std::set<std::array<index_t, 3>> t1_t2_valid_faces;

						for (const auto& f : faces_t1_valid)
							t1_t2_valid_faces.insert(make_face_key(f[0], f[1], f[2]));
						for (const auto& f : faces_t2_valid)
							t1_t2_valid_faces.insert(make_face_key(f[0], f[1], f[2]));

						// Step 2: For each T12, check if any of its 4 faces is shared with a valid T1/T2
						size_t n_t12 = global_t12_tets.size();
						std::vector<bool> touches_t1_t2(n_t12, false);

						for (size_t i = 0; i < n_t12; ++i) {
							for (int lf = 0; lf < 4; ++lf) {
								auto fk = make_face_key(
									global_t12_tets[i].faces[lf][0],
									global_t12_tets[i].faces[lf][1],
									global_t12_tets[i].faces[lf][2]);

								// Check if this face is shared with a valid T1/T2
								if (t1_t2_valid_faces.count(fk))
									touches_t1_t2[i] = true;
							}
						}

						// Step 3: Only keep T12s that directly touch valid T1/T2
						std::vector<bool> reachable(n_t12, false);

						for (size_t i = 0; i < n_t12; ++i) {
							if (touches_t1_t2[i]) {
								reachable[i] = true;
							}
						}

						// Step 4: Mark T12s that don't directly touch T1/T2 as isolated
						t12_isolated_count = 0;
						for (size_t i = 0; i < n_t12; ++i) {
							if (!reachable[i]) {
								global_t12_tets[i].fails_isolated = true;
								t12_isolated_count++;
							}
						}

						std::cout << "  Isolated filter:\n";
						std::cout << "    Valid T1/T2 faces in set: " << t1_t2_valid_faces.size() << "\n";
						int direct_touch = 0;
						for (size_t i = 0; i < n_t12; ++i) if (touches_t1_t2[i]) direct_touch++;
						std::cout << "    T12 directly touching T1/T2: " << direct_touch << "\n";
						std::cout << "    T12 isolated (no direct T1/T2): " << t12_isolated_count << "\n";

						// IMA rescue: isolated T12s with a vertex on the internal medial axis are rescued
						// Additional check: the IMA segment's midpoint must fall in solid on both L1 and L2
						int ima_rescued = 0;
						for (size_t i = 0; i < n_t12; ++i) {
						if (!global_t12_tets[i].fails_isolated) continue;
							if (global_t12_tets[i].fails_midpoint) continue;

							index_t vl1_r[2], vl2_r[2];
							int cl1_r = 0, cl2_r = 0;
							for (int lv = 0; lv < 4; ++lv) {
								index_t vi = global_t12_tets[i].verts[lv];
								if (vi < n_l1_pts) { if (cl1_r < 2) vl1_r[cl1_r++] = vi; }
								else { if (cl2_r < 2) vl2_r[cl2_r++] = vi; }
							}
							if (cl1_r != 2 || cl2_r != 2) continue;

							bool l1_has_ima = false;
							for (int k = 0; k < 2; ++k)
								if (vl1_r[k] >= global_ima_start_l1 && vl1_r[k] < global_ima_end_l1)
									l1_has_ima = true;
							bool l2_has_ima = false;
							for (int k = 0; k < 2; ++k)
								if (vl2_r[k] >= global_ima_start_l2 && vl2_r[k] < global_ima_end_l2)
									l2_has_ima = true;

							if (!l1_has_ima && !l2_has_ima) continue;

							bool rescue = false;
							if (l1_has_ima) {
								double mx = (coords_3d[3*vl1_r[0]] + coords_3d[3*vl1_r[1]]) * 0.5;
								double my = (coords_3d[3*vl1_r[0]+1] + coords_3d[3*vl1_r[1]+1]) * 0.5;
								bool in_l1 = sp_grid_l1.classify(mx, my) >= 0;
								bool in_l2 = sp_grid_l2.classify(mx, my) >= 0;
								if (in_l1 && in_l2) rescue = true;
							}
							if (!rescue && l2_has_ima) {
								double mx = (coords_3d[3*vl2_r[0]] + coords_3d[3*vl2_r[1]]) * 0.5;
								double my = (coords_3d[3*vl2_r[0]+1] + coords_3d[3*vl2_r[1]+1]) * 0.5;
								bool in_l1 = sp_grid_l1.classify(mx, my) >= 0;
								bool in_l2 = sp_grid_l2.classify(mx, my) >= 0;
								if (in_l1 && in_l2) rescue = true;
							}

							if (rescue) {
								global_t12_tets[i].fails_isolated = false;
								ima_rescued++;
								t12_isolated_count--;
							}
						}
						std::cout << "    T12 rescued by IMA vertex: " << ima_rescued << "\n";
					}

					std::cout << "  T12 diagnostic:\n";
					std::cout << "    Edge midpt not in solid: " << t12_edge_not_in_solid << "\n";
				}

				// Build vertex matrix (shared by all viewers)
				Eigen::MatrixXd V3d(N3d, 3);
				for (index_t i = 0; i < N3d; ++i) {
					V3d.row(i) << coords_3d[3 * i], coords_3d[3 * i + 1], coords_3d[3 * i + 2];
				}

				// Store T1/T2 valid faces globally BEFORE recompute_t12 (it needs them for isolated filter)
				global_V3d = V3d;
				global_faces_t1_valid = faces_t1_valid;
				global_faces_t2_valid = faces_t2_valid;
				global_faces_t1_removed = faces_t1_removed;
				global_faces_t2_removed = faces_t2_removed;

				// Compute T12 valid/removed from per-tet flags
				recompute_t12_from_filters();
				faces_t12_valid = global_faces_t12_valid;
				faces_t12_removed = global_faces_t12_removed;

				// =========================
				// ALGO 4: GROUP AFFINITY FILTER
				// Applied AFTER all standard filters. Removes tets whose base
				// and apex/opposite segment belong to different mapping groups.
				// =========================
				if ((recon_algo == 3 || recon_algo == 4) && !algo4_groups.empty()) {
					std::cout << "\n=== " << (recon_algo == 3 ? "ALGO3" : "ALGO4") << " Group Affinity Filter ===\n";

					auto find_group_l1 = [&](double x, double y) -> int {
						for (int gi = 0; gi < (int)algo4_groups.size(); ++gi)
							if (algo4_groups[gi].grid_l1.classify(x, y) >= 0) return gi;
						return -1;
					};
						auto find_group_l2 = [&](double x, double y) -> int {
							for (int gi = 0; gi < (int)algo4_groups.size(); ++gi)
								if (algo4_groups[gi].grid_l2.classify(x, y) >= 0) return gi;
							return -1;
						};

						// Extended group finders: try interior classification first, then boundary proximity.
						// Needed because contour vertices lie exactly ON the polygon edge and
						// ray-casting classify() returns -1 for them.
						auto find_group_with_boundary_l1 = [&](double x, double y) -> int {
							int g = find_group_l1(x, y);
							if (g >= 0) return g;
							Eigen::Vector2d p(x, y);
							for (int gi = 0; gi < (int)algo4_groups.size(); ++gi) {
								const auto& grp = algo4_groups[gi];
								if (grp.grid_l1.stored_contours &&
									point_near_contour_boundary(p, grp.refs_l1, grp.polys_l1, *grp.grid_l1.stored_contours))
									return gi;
							}
							return -1;
						};
						auto find_group_with_boundary_l2 = [&](double x, double y) -> int {
							int g = find_group_l2(x, y);
							if (g >= 0) return g;
							Eigen::Vector2d p(x, y);
							for (int gi = 0; gi < (int)algo4_groups.size(); ++gi) {
								const auto& grp = algo4_groups[gi];
								if (grp.grid_l2.stored_contours &&
									point_near_contour_boundary(p, grp.refs_l2, grp.polys_l2, *grp.grid_l2.stored_contours))
									return gi;
							}
							return -1;
						};

									// Group for T1 base: centroid of the base triangle (already verified inside solid).
									// Using only the centroid avoids the boundary issue with contour vertices.
									auto find_group_base_t1 = [&](const t1_t2_tet_info& info) -> int {
										return find_group_l1(info.centroid_x, info.centroid_y);
									};

									// Group for T2 base: centroid of the base triangle checked against L2.
									auto find_group_base_t2 = [&](const t1_t2_tet_info& info) -> int {
										return find_group_l2(info.centroid_x, info.centroid_y);
									};

									// Group for T12 L1 segment: check both vertices and the midpoint.
									// Contour vertices lie on the boundary so use the extended finder
									// that falls back to boundary proximity when classify() returns -1.
									// All non-negative results must agree; disagreement -> -1 (keep).
									auto find_group_seg_l1 = [&](index_t va, index_t vb) -> int {
										double ax = global_V3d(va, 0), ay = global_V3d(va, 1);
										double bx = global_V3d(vb, 0), by = global_V3d(vb, 1);
										double mx = (ax + bx) * 0.5, my = (ay + by) * 0.5;
										int ga = find_group_with_boundary_l1(ax, ay);
										int gb = find_group_with_boundary_l1(bx, by);
										int gm = find_group_with_boundary_l1(mx, my);
										int result = -1;
										for (int g : {ga, gb, gm}) {
											if (g < 0) continue;
											if (result < 0) result = g;
											else if (result != g) return -1;
										}
										return result;
									};

										// Group for T12 L2 segment: same approach as L1.
										auto find_group_seg_l2 = [&](index_t va, index_t vb) -> int {
											double ax = global_V3d(va, 0), ay = global_V3d(va, 1);
											double bx = global_V3d(vb, 0), by = global_V3d(vb, 1);
											double mx = (ax + bx) * 0.5, my = (ay + by) * 0.5;
											int ga = find_group_with_boundary_l2(ax, ay);
											int gb = find_group_with_boundary_l2(bx, by);
											int gm = find_group_with_boundary_l2(mx, my);
											int result = -1;
											for (int g : {ga, gb, gm}) {
												if (g < 0) continue;
												if (result < 0) result = g;
												else if (result != g) return -1;
											}
											return result;
										};

						// Check if a point is near the boundary of a specific group's L2 polygons.
						// Used to distinguish "boundary of same group" (keep) from "boundary of other group" (remove).
						auto near_group_boundary_l2 = [&](double x, double y, int gi) -> bool {
							if (gi < 0 || gi >= (int)algo4_groups.size()) return false;
							const auto& grp = algo4_groups[gi];
							if (!grp.grid_l2.stored_contours) return false;
							Eigen::Vector2d p(x, y);
							return point_near_contour_boundary(p, grp.refs_l2, grp.polys_l2, *grp.grid_l2.stored_contours);
						};

						// Check if a point is near the boundary of a specific group's L1 polygons.
						auto near_group_boundary_l1 = [&](double x, double y, int gi) -> bool {
							if (gi < 0 || gi >= (int)algo4_groups.size()) return false;
							const auto& grp = algo4_groups[gi];
							if (!grp.grid_l1.stored_contours) return false;
							Eigen::Vector2d p(x, y);
							return point_near_contour_boundary(p, grp.refs_l1, grp.polys_l1, *grp.grid_l1.stored_contours);
						};

						int rem_t1 = 0, rem_t2 = 0, rem_t12 = 0;

										// --- T1 filter ---
										// Base triangle centroid on L1 must be in same group as apex on L2
										global_faces_t1_valid.clear();
										global_faces_t1_removed.clear();
										for (auto& info : global_t1_tets) {
											bool already_invalid = !info.centroid_in_solid ||
												(global_filter_t1_midpoint && info.fails_midpoint);
											if (already_invalid) {
												for (int lf = 0; lf < 4; ++lf) global_faces_t1_removed.push_back(info.faces[lf]);
												continue;
											}
										// Apex must be in the same mapping group as the base (using extended boundary-aware finder).
										int g_base = find_group_base_t1(info);
										double ax = global_V3d(info.apex_vert, 0), ay = global_V3d(info.apex_vert, 1);
										bool do_remove = false;
										if (g_base >= 0) {
											int g_apex = find_group_with_boundary_l2(ax, ay);
											if (g_apex >= 0)
												do_remove = (g_apex != g_base);
											// g_apex == -1: completely outside all groups -> keep
										}
											if (do_remove) {
												info.fails_group_affinity = true;
												for (int lf = 0; lf < 4; ++lf) global_faces_t1_removed.push_back(info.faces[lf]);
												rem_t1++;
											} else {
												for (int lf = 0; lf < 4; ++lf) global_faces_t1_valid.push_back(info.faces[lf]);
											}
										}

										// --- T2 filter ---
										// Base triangle centroid on L2 must be in same group as apex on L1
										global_faces_t2_valid.clear();
										global_faces_t2_removed.clear();
										for (auto& info : global_t2_tets) {
						bool already_invalid = !info.centroid_in_solid ||
							(global_filter_t2_midpoint && info.fails_midpoint);
						if (already_invalid) {
							for (int lf = 0; lf < 4; ++lf) global_faces_t2_removed.push_back(info.faces[lf]);
							continue;
						}
										// Apex must be in the same mapping group as the base (using extended boundary-aware finder).
										int g_base = find_group_base_t2(info);
										double ax = global_V3d(info.apex_vert, 0), ay = global_V3d(info.apex_vert, 1);
										bool do_remove = false;
										if (g_base >= 0) {
											int g_apex = find_group_with_boundary_l1(ax, ay);
											if (g_apex >= 0)
												do_remove = (g_apex != g_base);
										}
											if (do_remove) {
												info.fails_group_affinity = true;
												for (int lf = 0; lf < 4; ++lf) global_faces_t2_removed.push_back(info.faces[lf]);
												rem_t2++;
											} else {
												for (int lf = 0; lf < 4; ++lf) global_faces_t2_valid.push_back(info.faces[lf]);
											}
										}

					// Re-run T12 isolated filter with the updated T1/T2 valid sets,
					// then apply group affinity to the surviving T12s
					recompute_t12_from_filters();

										// --- T12 filter ---
										// L1 segment midpoint must be in same group as L2 segment midpoint
										std::vector<std::array<index_t,3>> new_t12_valid, new_t12_removed;
										for (auto& info : global_t12_tets) {
											bool already_invalid = (global_filter_t12_midpoint && info.fails_midpoint) ||
												(global_filter_t12_isolated && info.fails_isolated);
											if (already_invalid) {
												for (int lf = 0; lf < 4; ++lf) new_t12_removed.push_back(info.faces[lf]);
												continue;
											}
						// Separate L1 and L2 vertices
						index_t vl1[2], vl2[2]; int cl1_c = 0, cl2_c = 0;
						for (int lv = 0; lv < 4; ++lv) {
							if (info.verts[lv] < n_l1_pts) { if (cl1_c < 2) vl1[cl1_c++] = info.verts[lv]; }
							else                          { if (cl2_c < 2) vl2[cl2_c++] = info.verts[lv]; }
						}
										// L1 segment group vs L2 segment group: if both are known and differ -> remove.
										// find_group_seg_l1/l2 use all 3 points (va, vb, midpoint) with boundary fallback,
										// so -1 means truly ambiguous (keep).
										int g1 = find_group_seg_l1(vl1[0], vl1[1]);
										int g2 = find_group_seg_l2(vl2[0], vl2[1]);
													bool do_remove_t12 = (g1 >= 0 && g2 >= 0 && g1 != g2);
													if (do_remove_t12) {
														info.fails_group_affinity = true;
														for (int lf = 0; lf < 4; ++lf) new_t12_removed.push_back(info.faces[lf]);
														rem_t12++;
													} else {
														for (int lf = 0; lf < 4; ++lf) new_t12_valid.push_back(info.faces[lf]);
													}
					}
					global_faces_t12_valid   = new_t12_valid;
					global_faces_t12_removed = new_t12_removed;

					// Update stats
					delaunay3d_T1_valid          = (int)global_faces_t1_valid.size()/4;
					delaunay3d_T1_valid_faces    = (int)global_faces_t1_valid.size();
					delaunay3d_T1_removed        = (int)global_faces_t1_removed.size()/4;
					delaunay3d_T1_removed_faces  = (int)global_faces_t1_removed.size();
					delaunay3d_T2_valid          = (int)global_faces_t2_valid.size()/4;
					delaunay3d_T2_valid_faces    = (int)global_faces_t2_valid.size();
					delaunay3d_T2_removed        = (int)global_faces_t2_removed.size()/4;
					delaunay3d_T2_removed_faces  = (int)global_faces_t2_removed.size();
					delaunay3d_T12_valid         = (int)global_faces_t12_valid.size()/4;
					delaunay3d_T12_valid_faces   = (int)global_faces_t12_valid.size();
					delaunay3d_T12_removed       = (int)global_faces_t12_removed.size()/4;
					delaunay3d_T12_removed_faces = (int)global_faces_t12_removed.size();
					delaunay3d_all_valid_faces   = (int)(global_faces_t1_valid.size()+
						global_faces_t2_valid.size()+global_faces_t12_valid.size());

					// Sync local copies for viewer setup below
					faces_t1_valid   = global_faces_t1_valid;
					faces_t1_removed = global_faces_t1_removed;
					faces_t2_valid   = global_faces_t2_valid;
					faces_t2_removed = global_faces_t2_removed;
					faces_t12_valid   = global_faces_t12_valid;
					faces_t12_removed = global_faces_t12_removed;

					std::cout << "  Group affinity removed: T1=" << rem_t1
						<< " T2=" << rem_t2 << " T12=" << rem_t12 << "\n";
					std::cout << "  After filter: T1v=" << delaunay3d_T1_valid
						<< " T2v=" << delaunay3d_T2_valid
						<< " T12v=" << delaunay3d_T12_valid << "\n";
				} // end algo 4 group affinity

				// All valid faces = T1 valid + T2 valid + T12 valid
				std::vector<std::array<index_t, 3>> faces_all_valid;
				faces_all_valid.insert(faces_all_valid.end(), faces_t1_valid.begin(), faces_t1_valid.end());
				faces_all_valid.insert(faces_all_valid.end(), faces_t2_valid.begin(), faces_t2_valid.end());
				faces_all_valid.insert(faces_all_valid.end(), faces_t12_valid.begin(), faces_t12_valid.end());
				delaunay3d_all_valid_faces = static_cast<int>(faces_all_valid.size());

				std::cout << "\nFiltering:\n";
				std::cout << "  T1 valid:   " << delaunay3d_T1_valid << " tets, " << delaunay3d_T1_valid_faces << " faces\n";
				std::cout << "  T1 removed: " << delaunay3d_T1_removed << " tets, " << delaunay3d_T1_removed_faces << " faces";
				if (global_t1_removed_by_midpoint > 0)
					std::cout << " (edge-midpoint: " << global_t1_removed_by_midpoint << ")";
				std::cout << "\n";
				std::cout << "  T2 valid:   " << delaunay3d_T2_valid << " tets, " << delaunay3d_T2_valid_faces << " faces\n";
				std::cout << "  T2 removed: " << delaunay3d_T2_removed << " tets, " << delaunay3d_T2_removed_faces << " faces";
				if (global_t2_removed_by_midpoint > 0)
					std::cout << " (edge-midpoint: " << global_t2_removed_by_midpoint << ")";
				std::cout << "\n";
				std::cout << "  T12 valid:  " << delaunay3d_T12_valid << " tets, " << delaunay3d_T12_valid_faces << " faces\n";
				std::cout << "  T12 removed:" << delaunay3d_T12_removed << " tets, " << delaunay3d_T12_removed_faces << " faces\n";
				std::cout << "  All valid:  " << delaunay3d_all_valid_faces << " faces (T1v+T2v+T12v)\n";

				// Helper to set up a viewer with flat color
				auto setup_tet_viewer = [&](igl::opengl::glfw::Viewer& vw,
					const std::vector<std::array<index_t, 3>>& faces,
					const Eigen::RowVector3d& color) {
					if (!faces.empty()) {
						int nf = static_cast<int>(faces.size());
						Eigen::MatrixXi F(nf, 3);
						for (int i = 0; i < nf; ++i)
							F.row(i) << static_cast<int>(faces[i][0]),
								static_cast<int>(faces[i][1]),
								static_cast<int>(faces[i][2]);
						vw.data().set_mesh(V3d, F);
						vw.data().show_lines = true;
						vw.data().show_faces = true;
						Eigen::MatrixXd C(nf, 3);
						for (int i = 0; i < nf; ++i)
							C.row(i) = color;
						vw.data().set_colors(C);
					} else {
						vw.data().set_mesh(Vbox, Fbox);
						vw.data().show_faces = false;
					}
				};

				// Helper to set up a viewer with per-type coloring for all-valid
				auto setup_all_valid_viewer = [&](igl::opengl::glfw::Viewer& vw,
					const std::vector<std::array<index_t, 3>>& f_t1v,
					const std::vector<std::array<index_t, 3>>& f_t2v,
					const std::vector<std::array<index_t, 3>>& f_t12) {
					std::vector<std::array<index_t, 3>> all_f;
					all_f.insert(all_f.end(), f_t1v.begin(), f_t1v.end());
					all_f.insert(all_f.end(), f_t2v.begin(), f_t2v.end());
					all_f.insert(all_f.end(), f_t12.begin(), f_t12.end());
					if (!all_f.empty()) {
						int nf = static_cast<int>(all_f.size());
						Eigen::MatrixXi F(nf, 3);
						for (int i = 0; i < nf; ++i)
							F.row(i) << static_cast<int>(all_f[i][0]),
								static_cast<int>(all_f[i][1]),
								static_cast<int>(all_f[i][2]);
						vw.data().set_mesh(V3d, F);
						vw.data().show_lines = true;
						vw.data().show_faces = true;
						Eigen::MatrixXd C(nf, 3);
						int offset = 0;
						for (int i = 0; i < (int)f_t1v.size(); ++i)
							C.row(offset++) << 0.4, 0.6, 1.0;
						for (int i = 0; i < (int)f_t2v.size(); ++i)
							C.row(offset++) << 0.4, 1.0, 0.4;
						for (int i = 0; i < (int)f_t12.size(); ++i)
							C.row(offset++) << 1.0, 0.7, 0.3;
						vw.data().set_colors(C);
					} else {
						vw.data().set_mesh(Vbox, Fbox);
						vw.data().show_faces = false;
					}
				};

				// Main Delaunay viewer: boundary faces colored by tet type
				{
					std::vector<std::array<index_t, 3>> bf_ordered;
					std::vector<int> bf_type; // 1=T1, 2=T2, 12=T12, 0=degenerate

					for (index_t t = 0; t < nb_tets; ++t) {
						const signed_index_t* v = delaunay3d->cell_to_v() + 4 * t;
						int count_l1 = 0;
						for (int lv = 0; lv < 4; ++lv)
							if (static_cast<index_t>(v[lv]) < n_l1_pts) count_l1++;
						int tet_type = 0;
						if (count_l1 == 3) tet_type = 1;
						else if (count_l1 == 1) tet_type = 2;
						else if (count_l1 == 2) tet_type = 12;

						for (index_t lf = 0; lf < 4; ++lf) {
							signed_index_t adj = delaunay3d->cell_adjacent(t, lf);
							if (adj == NO_INDEX) {
								bf_ordered.push_back({
									static_cast<index_t>(v[face_vertex[lf][0]]),
									static_cast<index_t>(v[face_vertex[lf][1]]),
									static_cast<index_t>(v[face_vertex[lf][2]])
								});
								bf_type.push_back(tet_type);
							}
						}
					}

					int nbf = static_cast<int>(bf_ordered.size());
					Eigen::MatrixXi F3d(nbf, 3);
					for (int i = 0; i < nbf; ++i)
						F3d.row(i) << static_cast<int>(bf_ordered[i][0]),
							static_cast<int>(bf_ordered[i][1]),
							static_cast<int>(bf_ordered[i][2]);

					viewer_delaunay3d.data().set_mesh(V3d, F3d);
					viewer_delaunay3d.data().show_lines = true;
					viewer_delaunay3d.data().show_faces = true;

					Eigen::MatrixXd C3d(nbf, 3);
					for (int i = 0; i < nbf; ++i) {
						if (bf_type[i] == 1)       C3d.row(i) << 0.4, 0.6, 1.0;
						else if (bf_type[i] == 2)  C3d.row(i) << 0.4, 1.0, 0.4;
						else if (bf_type[i] == 12) C3d.row(i) << 1.0, 0.7, 0.3;
						else                       C3d.row(i) << 0.6, 0.6, 0.6;
					}
					viewer_delaunay3d.data().set_colors(C3d);
				}

				// Store remaining face data globally for interactive viewer
				// (global_V3d, global_faces_t1_valid/removed, global_faces_t2_valid/removed
				//  were already set before recompute_t12_from_filters)
				global_faces_t12_valid = faces_t12_valid;
				global_faces_t12_removed = faces_t12_removed;
				interactive_needs_rebuild = true;

				// T1 viewer: blue (all T1)
				setup_tet_viewer(viewer_t1, faces_t1, Eigen::RowVector3d(0.4, 0.6, 1.0));
				// T2 viewer: green (all T2)
				setup_tet_viewer(viewer_t2, faces_t2, Eigen::RowVector3d(0.4, 1.0, 0.4));
				// T12 viewer: orange
				setup_tet_viewer(viewer_t12, faces_t12, Eigen::RowVector3d(1.0, 0.7, 0.3));

				// Filtered viewers
				setup_tet_viewer(viewer_t1_valid, faces_t1_valid, Eigen::RowVector3d(0.4, 0.6, 1.0));
				setup_tet_viewer(viewer_t1_removed, faces_t1_removed, Eigen::RowVector3d(0.8, 0.2, 0.2));
				setup_tet_viewer(viewer_t2_valid, faces_t2_valid, Eigen::RowVector3d(0.4, 1.0, 0.4));
				setup_tet_viewer(viewer_t2_removed, faces_t2_removed, Eigen::RowVector3d(0.8, 0.2, 0.2));
				setup_tet_viewer(viewer_t12_valid, faces_t12_valid, Eigen::RowVector3d(1.0, 0.7, 0.3));
				setup_tet_viewer(viewer_t12_removed, faces_t12_removed, Eigen::RowVector3d(0.8, 0.2, 0.2));
				setup_all_valid_viewer(viewer_all_valid, faces_t1_valid, faces_t2_valid, faces_t12_valid);

				// All valid gray: uniform gray for all valid tets
				setup_tet_viewer(viewer_all_valid_gray, faces_all_valid, Eigen::RowVector3d(0.7, 0.7, 0.7));

				// Free boundary using igl::boundary_facets
				// Build T_all directly from stored tet vertices (not reconstructed from faces)
				{
				// Count valid tets using the SAME conditions as the T_all fill loops below
					int n_valid_t1 = 0, n_valid_t2 = 0, n_valid_t12 = 0;
					for (const auto& info : global_t1_tets)
						if (info.centroid_in_solid && !(global_filter_t1_midpoint && info.fails_midpoint))
							n_valid_t1++;
					for (const auto& info : global_t2_tets)
						if (info.centroid_in_solid && !(global_filter_t2_midpoint && info.fails_midpoint))
							n_valid_t2++;
					// Count T12 with the same conditions as the fill loop (not from face list)
					// to avoid size mismatch when the affinity filter has updated face lists
					// but not the per-tet flags.
					for (const auto& info : global_t12_tets) {
						if (global_filter_t12_midpoint && info.fails_midpoint) continue;
						if (global_filter_t12_isolated && info.fails_isolated) continue;
						if (info.fails_group_affinity) continue;
						n_valid_t12++;
					}

					// Build T_all from original vertex order + T12 from verts[]
					int n_all_tets = n_valid_t1 + n_valid_t2 + n_valid_t12;
					Eigen::MatrixXi T_all(n_all_tets, 4);
					int ti = 0;

								for (const auto& info : global_t1_tets) {
									if (!info.centroid_in_solid) continue;
									if (global_filter_t1_midpoint && info.fails_midpoint) continue;
									if (info.fails_group_affinity) continue;
									for (int lv = 0; lv < 4; ++lv)
										T_all(ti, lv) = static_cast<int>(info.orig_verts[lv]);
									ti++;
								}
								for (const auto& info : global_t2_tets) {
									if (!info.centroid_in_solid) continue;
									if (global_filter_t2_midpoint && info.fails_midpoint) continue;
									if (info.fails_group_affinity) continue;
									for (int lv = 0; lv < 4; ++lv)
										T_all(ti, lv) = static_cast<int>(info.orig_verts[lv]);
									ti++;
								}
					// T12: iterate global_t12_tets and pick valid ones
					for (size_t i = 0; i < global_t12_tets.size(); ++i) {
						const auto& info = global_t12_tets[i];
						bool valid = true;
						if (global_filter_t12_midpoint && info.fails_midpoint) valid = false;
						if (valid && global_filter_t12_isolated && info.fails_isolated) valid = false;
						if (valid && info.fails_group_affinity) valid = false;
						if (!valid) continue;
						for (int lv = 0; lv < 4; ++lv)
							T_all(ti, lv) = static_cast<int>(info.verts[lv]);
						ti++;
					}

					std::cout << "  Tets for boundary_facets: " << ti
						<< " (T1v=" << n_valid_t1 << " T2v=" << n_valid_t2 << " T12v=" << n_valid_t12 << ")\n";

					// Orient all tets positively: if det < 0, swap verts 0 and 1
					int reoriented = 0;
					for (int r = 0; r < ti; ++r) {
						int i0 = T_all(r, 0), i1 = T_all(r, 1), i2 = T_all(r, 2), i3 = T_all(r, 3);
						Eigen::Vector3d p0 = V3d.row(i0);
						Eigen::Vector3d p1 = V3d.row(i1);
						Eigen::Vector3d p2 = V3d.row(i2);
						Eigen::Vector3d p3 = V3d.row(i3);
						double det = (p1 - p0).dot((p2 - p0).cross(p3 - p0));
						if (det < 0) {
							std::swap(T_all(r, 0), T_all(r, 1));
							reoriented++;
						}
					}
					std::cout << "  Reoriented tets (negative det): " << reoriented << " / " << ti << "\n";

					T_all.conservativeResize(ti, 4);

					Eigen::MatrixXi F_fb;
					igl::boundary_facets(T_all, F_fb);

					global_faces_free_boundary.clear();
					for (int i = 0; i < F_fb.rows(); ++i)
						global_faces_free_boundary.push_back({static_cast<index_t>(F_fb(i,0)), static_cast<index_t>(F_fb(i,1)), static_cast<index_t>(F_fb(i,2))});
					delaunay3d_free_boundary_faces = static_cast<int>(F_fb.rows());

					std::cout << "  Free boundary: " << delaunay3d_free_boundary_faces << " faces\n";

					// Reindex vertices for manifold check and OBJ export
					std::map<int, int> old_to_new;
					int next_id = 0;
					Eigen::MatrixXi F_fb_reindexed(delaunay3d_free_boundary_faces, 3);
					for (int i = 0; i < delaunay3d_free_boundary_faces; ++i) {
						for (int lv = 0; lv < 3; ++lv) {
							int old_v = F_fb(i, lv);
							if (old_to_new.find(old_v) == old_to_new.end())
								old_to_new[old_v] = next_id++;
							F_fb_reindexed(i, lv) = old_to_new[old_v];
						}
					}
					Eigen::MatrixXd V_fb(next_id, 3);
					for (const auto& kv : old_to_new)
						V_fb.row(kv.second) = V3d.row(kv.first);

					// Orient faces outward: flip faces whose normal points toward the centroid
					{
						Eigen::Vector3d centroid = V_fb.colwise().mean();
						int flipped = 0;
						for (int i = 0; i < F_fb_reindexed.rows(); ++i) {
							Eigen::Vector3d a = V_fb.row(F_fb_reindexed(i, 0));
							Eigen::Vector3d b = V_fb.row(F_fb_reindexed(i, 1));
							Eigen::Vector3d c = V_fb.row(F_fb_reindexed(i, 2));
							Eigen::Vector3d n = (b - a).cross(c - a);
							Eigen::Vector3d face_center = (a + b + c) / 3.0;
							if (n.dot(face_center - centroid) < 0) {
								std::swap(F_fb_reindexed(i, 0), F_fb_reindexed(i, 1));
								flipped++;
							}
						}
						std::cout << "  Faces oriented outward: flipped " << flipped << " / " << F_fb_reindexed.rows() << "\n";
					}

					// Also update global_faces_free_boundary to match the reoriented faces
					{
						std::map<int, int> new_to_old;
						for (const auto& kv : old_to_new)
							new_to_old[kv.second] = kv.first;
						for (int i = 0; i < F_fb_reindexed.rows(); ++i) {
							global_faces_free_boundary[i] = {
								static_cast<index_t>(new_to_old[F_fb_reindexed(i, 0)]),
								static_cast<index_t>(new_to_old[F_fb_reindexed(i, 1)]),
								static_cast<index_t>(new_to_old[F_fb_reindexed(i, 2)])
							};
						}
					}

					global_manifold_info = check_manifold(F_fb_reindexed);

					std::cout << "\n  === MANIFOLD DIAGNOSTIC ===\n";
					std::cout << "  Edge-manifold:   " << (global_manifold_info.edge_manifold ? "YES" : "NO") << "\n";
					std::cout << "  Vertex-manifold: " << (global_manifold_info.vertex_manifold ? "YES" : "NO") << "\n";
					std::cout << "  Euler characteristic: " << global_manifold_info.euler_characteristic << "\n";
					if (!global_manifold_info.vertex_manifold)
						std::cout << "  Non-manifold vertices: " << global_manifold_info.non_manifold_vertices << "\n";
					if (!global_manifold_info.edge_manifold)
						std::cout << "  Non-manifold edges: " << global_manifold_info.non_manifold_edges << "\n";
					std::cout << "  Boundary vertices: " << next_id << "\n";
					std::cout << "  Result: " << (global_manifold_info.is_manifold() ? "MANIFOLD" : "NOT MANIFOLD") << "\n";
					std::cout << "  =========================\n";

					// =========================
					// GAP FILL: create new T12 tets to close holes caused by IMA apex vertices
					// Strategy:
					//   1. Find non-manifold edges where one vertex is on the contour and the other is an internal (IMA/EMA) point
					//   2. Find the 2 faces sharing that edge; their 3rd vertices are on the opposite level's contour
					//   3. Walk the SHORT path along that opposite-level contour between those 2 vertices (only contour vertices)
					//   4. Create T12 tets: non-manifold edge segment x each contour segment along the short path
					// =========================
					if (!global_manifold_info.is_manifold() && !global_manifold_info.non_manifold_edge_list.empty()) {
						std::cout << "\n  === GAP FILL REPAIR ===\n";

						// Build reverse map: reindexed vertex -> original vertex
						std::map<int, int> new_to_old_repair;
						for (const auto& kv : old_to_new)
							new_to_old_repair[kv.second] = kv.first;

						// Build contour vertex lookup tables
						const auto& contours_src_l1 = (resampleFactor > 1) ? resampled_l1 : levels[startIdx - 1].contours;
						const auto& contours_src_l2 = (resampleFactor > 1) ? resampled_l2 : levels[startIdx].contours;

						// Build per-contour offset tables
						// contour_offsets_lX[c] = first global index of contour c
						std::vector<int> contour_offsets_l1;
						{
							int offset = 0;
							for (const auto& c : contours_src_l1) {
								contour_offsets_l1.push_back(offset);
								offset += static_cast<int>(c.points.size());
							}
							contour_offsets_l1.push_back(offset);
						}
						std::vector<int> contour_offsets_l2;
						{
							int offset = 0;
							for (const auto& c : contours_src_l2) {
								contour_offsets_l2.push_back(offset);
								offset += static_cast<int>(c.points.size());
							}
							contour_offsets_l2.push_back(offset);
						}

						// Helper: find which contour a vertex belongs to and its local index
						// Returns {contour_id, local_index, contour_size} or {-1,-1,-1}
						auto find_contour_info_l1 = [&](int global_idx) -> std::tuple<int,int,int> {
							if (global_idx < 0 || global_idx >= aug_contour_l1)
								return {-1, -1, -1};
							for (int c = 0; c < (int)contour_offsets_l1.size() - 1; ++c) {
								if (global_idx >= contour_offsets_l1[c] && global_idx < contour_offsets_l1[c+1]) {
									int local = global_idx - contour_offsets_l1[c];
									int sz = contour_offsets_l1[c+1] - contour_offsets_l1[c];
									return {c, local, sz};
								}
							}
							return {-1, -1, -1};
						};
						auto find_contour_info_l2 = [&](int global_idx) -> std::tuple<int,int,int> {
							int local_idx = global_idx - static_cast<int>(n_l1_pts);
							if (local_idx < 0 || local_idx >= aug_contour_l2)
								return {-1, -1, -1};
							for (int c = 0; c < (int)contour_offsets_l2.size() - 1; ++c) {
								if (local_idx >= contour_offsets_l2[c] && local_idx < contour_offsets_l2[c+1]) {
									int local = local_idx - contour_offsets_l2[c];
									int sz = contour_offsets_l2[c+1] - contour_offsets_l2[c];
									return {c, local, sz};
								}
							}
							return {-1, -1, -1};
						};

						// Classification helpers
						auto is_contour_l1 = [&](int idx) { return idx >= 0 && idx < aug_contour_l1; };
						auto is_contour_l2 = [&](int idx) { return idx >= (int)n_l1_pts && idx < (int)n_l1_pts + aug_contour_l2; };
						auto is_internal_l1 = [&](int idx) { return idx >= aug_contour_l1 && idx < (int)n_l1_pts; };
						auto is_internal_l2 = [&](int idx) { return idx >= (int)n_l1_pts + aug_contour_l2 && idx < (int)N3d; };

						// Build edge-to-faces map from free boundary (original indices)
						std::map<std::pair<int,int>, std::vector<int>> edge_to_fb_faces;
						for (int fi = 0; fi < (int)global_faces_free_boundary.size(); ++fi) {
							int fv[3] = {
								static_cast<int>(global_faces_free_boundary[fi][0]),
								static_cast<int>(global_faces_free_boundary[fi][1]),
								static_cast<int>(global_faces_free_boundary[fi][2])
							};
							for (int e = 0; e < 3; ++e) {
								int ea = fv[e], eb = fv[(e+1)%3];
								if (ea > eb) std::swap(ea, eb);
								edge_to_fb_faces[{ea, eb}].push_back(fi);
							}
						}

						std::vector<std::array<index_t, 4>> repair_tets;
						int nm_processed = 0, nm_skipped = 0;

						std::cout << "  Non-manifold edges to analyze: " << global_manifold_info.non_manifold_edge_list.size() << "\n";
						std::cout << "  Index ranges: L1 contour=[0.." << aug_contour_l1-1
							<< "] L1 internal=[" << aug_contour_l1 << ".." << (int)n_l1_pts-1
							<< "] L2 contour=[" << (int)n_l1_pts << ".." << (int)n_l1_pts+aug_contour_l2-1
							<< "] L2 internal=[" << (int)n_l1_pts+aug_contour_l2 << ".." << (int)N3d-1 << "]\n";

						for (const auto& nm_edge_reindexed : global_manifold_info.non_manifold_edge_list) {
							int orig_a = new_to_old_repair[nm_edge_reindexed.first];
							int orig_b = new_to_old_repair[nm_edge_reindexed.second];

							// Classify each vertex for debug
							auto classify = [&](int idx) -> std::string {
								if (is_contour_l1(idx)) return "L1_contour";
								if (is_internal_l1(idx)) return "L1_internal";
								if (is_contour_l2(idx)) return "L2_contour";
								if (is_internal_l2(idx)) return "L2_internal";
								return "UNKNOWN";
							};
							std::cout << "  Analyzing NM edge: reindexed[" << nm_edge_reindexed.first << "," << nm_edge_reindexed.second
								<< "] -> orig[" << orig_a << "," << orig_b << "]"
								<< " types: " << classify(orig_a) << " + " << classify(orig_b) << "\n";

							// Determine the edge type for repair strategy
							int contour_v = -1, internal_v = -1;
							bool nm_on_l1 = false, nm_on_l2 = false;
							bool nm_l1l1  = false, nm_l2l2  = false;

							if (is_contour_l1(orig_a) && is_internal_l1(orig_b)) {
								contour_v = orig_a; internal_v = orig_b; nm_on_l1 = true;
							} else if (is_contour_l1(orig_b) && is_internal_l1(orig_a)) {
								contour_v = orig_b; internal_v = orig_a; nm_on_l1 = true;
							} else if (is_contour_l2(orig_a) && is_internal_l2(orig_b)) {
								contour_v = orig_a; internal_v = orig_b; nm_on_l2 = true;
							} else if (is_contour_l2(orig_b) && is_internal_l2(orig_a)) {
								contour_v = orig_b; internal_v = orig_a; nm_on_l2 = true;
							} else if (is_contour_l1(orig_a) && is_contour_l1(orig_b)) {
								// Both L1 contour vertices: gap closed by L2 contour path
								nm_l1l1 = true;
							} else if (is_contour_l2(orig_a) && is_contour_l2(orig_b)) {
								// Both L2 contour vertices: gap closed by L1 contour path
								nm_l2l2 = true;
							} else {
								nm_skipped++;
								continue;
							}

							// Find the 2+ faces that share this non-manifold edge
							int ea = orig_a, eb = orig_b;
							if (ea > eb) std::swap(ea, eb);
							auto it_edge = edge_to_fb_faces.find({ea, eb});
							if (it_edge == edge_to_fb_faces.end() || it_edge->second.size() < 2) {
								nm_skipped++;
								continue;
							}

							// Collect the 3rd vertices of faces sharing this edge
							// Filter: only those that are contour vertices on the OPPOSITE level
							std::vector<int> opposite_verts;
							for (int fi : it_edge->second) {
								int fv[3] = {
									static_cast<int>(global_faces_free_boundary[fi][0]),
									static_cast<int>(global_faces_free_boundary[fi][1]),
									static_cast<int>(global_faces_free_boundary[fi][2])
								};
								for (int k = 0; k < 3; ++k) {
									if (fv[k] != orig_a && fv[k] != orig_b) {
										// Only accept if this vertex is a contour vertex on the opposite level
										bool accept = false;
										if ((nm_on_l1 || nm_l1l1) && is_contour_l2(fv[k])) accept = true;
										if ((nm_on_l2 || nm_l2l2) && is_contour_l1(fv[k])) accept = true;
										if (accept) {
											// Avoid duplicates
											bool dup = false;
											for (int ov : opposite_verts) if (ov == fv[k]) { dup = true; break; }
											if (!dup) opposite_verts.push_back(fv[k]);
										}
										break;
									}
								}
							}

							if (opposite_verts.size() < 2) { nm_skipped++; continue; }

							int opp_v0 = opposite_verts[0];
							int opp_v1 = opposite_verts[1];

							int opp_contour_id, opp_local_0, opp_sz, opp_local_1;

							bool opp_on_l2 = nm_on_l1 || nm_l1l1;
							if (opp_on_l2) {
								auto [c0, l0, s0] = find_contour_info_l2(opp_v0);
								auto [c1, l1, s1] = find_contour_info_l2(opp_v1);
								if (c0 < 0 || c1 < 0 || c0 != c1) { nm_skipped++; continue; }
								opp_contour_id = c0; opp_local_0 = l0; opp_sz = s0; opp_local_1 = l1;
							} else {
								auto [c0, l0, s0] = find_contour_info_l1(opp_v0);
								auto [c1, l1, s1] = find_contour_info_l1(opp_v1);
								if (c0 < 0 || c1 < 0 || c0 != c1) { nm_skipped++; continue; }
								opp_contour_id = c0; opp_local_0 = l0; opp_sz = s0; opp_local_1 = l1;
							}

							// Find the SHORT path between them along the closed contour
							int contour_size = opp_sz;
							int idx_a = opp_local_0;
							int idx_b = opp_local_1;

							int dist_forward  = (idx_b - idx_a + contour_size) % contour_size;
							int dist_backward = (idx_a - idx_b + contour_size) % contour_size;

							int step, path_len;
							if (dist_forward <= dist_backward) {
								step = 1;  path_len = dist_forward;
							} else {
								step = -1; path_len = dist_backward;
							}

							if (path_len < 1) { nm_skipped++; continue; }

							// Compute global base offset for this contour
							int contour_global_base;
							if (opp_on_l2) {
								contour_global_base = static_cast<int>(n_l1_pts) + contour_offsets_l2[opp_contour_id];
							} else {
								contour_global_base = contour_offsets_l1[opp_contour_id];
							}

							{
								std::string nm_type = nm_l1l1 ? "L1-L1" : nm_l2l2 ? "L2-L2" : nm_on_l1 ? "L1" : "L2";
								std::cout << "  NM edge [" << orig_a << "," << orig_b << "] type=" << nm_type;
								if (!nm_l1l1 && !nm_l2l2)
									std::cout << " contour_v=" << contour_v << " internal_v=" << internal_v;
								std::cout
									<< " | opp verts: " << opp_v0 << "(local " << idx_a << ")"
									<< " -> " << opp_v1 << "(local " << idx_b << ")"
									<< " path_len=" << path_len
									<< " contour_size=" << contour_size << "\n";
							}

							// Create T12 tets: non-manifold edge x each contour segment along the short path
							for (int s = 0; s < path_len; ++s) {
								int local_from = (idx_a + s * step + contour_size) % contour_size;
								int local_to   = (idx_a + (s + 1) * step + contour_size) % contour_size;
								int seg_v0 = contour_global_base + local_from;
								int seg_v1 = contour_global_base + local_to;

										if (nm_l1l1 || nm_l2l2) {
											// Both NM vertices are contour verts; T12-like tet
											repair_tets.push_back({
												static_cast<index_t>(orig_a),
												static_cast<index_t>(orig_b),
												static_cast<index_t>(seg_v0),
												static_cast<index_t>(seg_v1)
											});
										} else {
											repair_tets.push_back({
												static_cast<index_t>(contour_v),
												static_cast<index_t>(internal_v),
												static_cast<index_t>(seg_v0),
												static_cast<index_t>(seg_v1)
											});
										}
							}

							nm_processed++;
						}

						std::cout << "  NM edges processed: " << nm_processed
							<< ", skipped: " << nm_skipped
							<< ", repair tets created: " << repair_tets.size() << "\n";

						if (!repair_tets.empty()) {
							// Add repair tets to T_all
							int old_count = T_all.rows();
							int new_count = old_count + static_cast<int>(repair_tets.size());
							T_all.conservativeResize(new_count, 4);
							for (size_t ri = 0; ri < repair_tets.size(); ++ri) {
								int row = old_count + static_cast<int>(ri);
								for (int lv = 0; lv < 4; ++lv)
									T_all(row, lv) = static_cast<int>(repair_tets[ri][lv]);
							}

							// Orient new tets positively
							int reoriented_repair = 0;
							for (int r = old_count; r < new_count; ++r) {
								int i0 = T_all(r, 0), i1 = T_all(r, 1), i2 = T_all(r, 2), i3 = T_all(r, 3);
								Eigen::Vector3d p0 = V3d.row(i0);
								Eigen::Vector3d p1 = V3d.row(i1);
								Eigen::Vector3d p2 = V3d.row(i2);
								Eigen::Vector3d p3 = V3d.row(i3);
								double det = (p1 - p0).dot((p2 - p0).cross(p3 - p0));
								if (det < 0) {
									std::swap(T_all(r, 0), T_all(r, 1));
									reoriented_repair++;
								}
							}
							std::cout << "  Reoriented repair tets: " << reoriented_repair << "\n";

							// Recompute boundary_facets
							igl::boundary_facets(T_all, F_fb);

							global_faces_free_boundary.clear();
							for (int i = 0; i < F_fb.rows(); ++i)
								global_faces_free_boundary.push_back({
									static_cast<index_t>(F_fb(i,0)),
									static_cast<index_t>(F_fb(i,1)),
									static_cast<index_t>(F_fb(i,2))
								});
							delaunay3d_free_boundary_faces = static_cast<int>(F_fb.rows());

							// Reindex vertices
							old_to_new.clear();
							next_id = 0;
							F_fb_reindexed.resize(delaunay3d_free_boundary_faces, 3);
							for (int i = 0; i < delaunay3d_free_boundary_faces; ++i) {
								for (int lv = 0; lv < 3; ++lv) {
									int old_v = F_fb(i, lv);
									if (old_to_new.find(old_v) == old_to_new.end())
										old_to_new[old_v] = next_id++;
									F_fb_reindexed(i, lv) = old_to_new[old_v];
								}
							}
							V_fb.resize(next_id, 3);
							for (const auto& kv : old_to_new)
								V_fb.row(kv.second) = V3d.row(kv.first);

							// Orient faces outward
							{
								Eigen::Vector3d centroid = V_fb.colwise().mean();
								int flipped = 0;
								for (int i = 0; i < F_fb_reindexed.rows(); ++i) {
									Eigen::Vector3d a = V_fb.row(F_fb_reindexed(i, 0));
									Eigen::Vector3d b = V_fb.row(F_fb_reindexed(i, 1));
									Eigen::Vector3d c = V_fb.row(F_fb_reindexed(i, 2));
									Eigen::Vector3d n = (b - a).cross(c - a);
									Eigen::Vector3d face_center = (a + b + c) / 3.0;
									if (n.dot(face_center - centroid) < 0) {
										std::swap(F_fb_reindexed(i, 0), F_fb_reindexed(i, 1));
										flipped++;
									}
								}
								std::cout << "  Faces oriented outward (repair): flipped " << flipped << " / " << F_fb_reindexed.rows() << "\n";
							}

							// Update global_faces_free_boundary to match reoriented faces
							{
								std::map<int, int> new_to_old_r;
								for (const auto& kv : old_to_new)
									new_to_old_r[kv.second] = kv.first;
								for (int i = 0; i < F_fb_reindexed.rows(); ++i) {
									global_faces_free_boundary[i] = {
										static_cast<index_t>(new_to_old_r[F_fb_reindexed(i, 0)]),
										static_cast<index_t>(new_to_old_r[F_fb_reindexed(i, 1)]),
										static_cast<index_t>(new_to_old_r[F_fb_reindexed(i, 2)])
									};
								}
							}

							// Re-check manifold
							global_manifold_info = check_manifold(F_fb_reindexed);

							std::cout << "\n  === MANIFOLD AFTER REPAIR ===\n";
							std::cout << "  Free boundary faces: " << delaunay3d_free_boundary_faces << "\n";
							std::cout << "  Edge-manifold:   " << (global_manifold_info.edge_manifold ? "YES" : "NO") << "\n";
							std::cout << "  Vertex-manifold: " << (global_manifold_info.vertex_manifold ? "YES" : "NO") << "\n";
							std::cout << "  Euler characteristic: " << global_manifold_info.euler_characteristic << "\n";
							if (!global_manifold_info.vertex_manifold)
								std::cout << "  Non-manifold vertices: " << global_manifold_info.non_manifold_vertices << "\n";
							if (!global_manifold_info.edge_manifold)
								std::cout << "  Non-manifold edges: " << global_manifold_info.non_manifold_edges << "\n";
							std::cout << "  Result: " << (global_manifold_info.is_manifold() ? "MANIFOLD" : "NOT MANIFOLD") << "\n";
							std::cout << "  =========================\n";
						}

						std::cout << "  === END GAP FILL ===\n";
					}

					setup_tet_viewer(viewer_free_boundary, global_faces_free_boundary, Eigen::RowVector3d(0.7, 0.7, 0.7));

					{
						std::string obj_path = "C:/Users/DELL/Desktop/Data/free_boundary.obj";
						if (igl::writeOBJ(obj_path, V_fb, F_fb_reindexed))
							std::cout << "  OBJ exported: " << obj_path << " ("
								<< V_fb.rows() << " vertices, " << F_fb_reindexed.rows() << " faces)\n";
						else
							std::cerr << "  ERROR: Could not write OBJ to " << obj_path << "\n";
					}
				}

				// Interactive viewer: initial setup with all valid
				setup_all_valid_viewer(viewer_interactive, faces_t1_valid, faces_t2_valid, faces_t12_valid);
			}
			else {
				std::cout << "Not enough points for 3D Delaunay (need >= 4)\n";
				viewer_delaunay3d.data().set_mesh(Vbox, Fbox);
				viewer_delaunay3d.data().show_faces = false;
				viewer_t1.data().set_mesh(Vbox, Fbox);
				viewer_t1.data().show_faces = false;
				viewer_t2.data().set_mesh(Vbox, Fbox);
				viewer_t2.data().show_faces = false;
				viewer_t12.data().set_mesh(Vbox, Fbox);
				viewer_t12.data().show_faces = false;
				viewer_t1_valid.data().set_mesh(Vbox, Fbox);
				viewer_t1_valid.data().show_faces = false;
				viewer_t1_removed.data().set_mesh(Vbox, Fbox);
				viewer_t1_removed.data().show_faces = false;
				viewer_t2_valid.data().set_mesh(Vbox, Fbox);
				viewer_t2_valid.data().show_faces = false;
				viewer_t2_removed.data().set_mesh(Vbox, Fbox);
				viewer_t2_removed.data().show_faces = false;
				viewer_t12_valid.data().set_mesh(Vbox, Fbox);
				viewer_t12_valid.data().show_faces = false;
				viewer_t12_removed.data().set_mesh(Vbox, Fbox);
				viewer_t12_removed.data().show_faces = false;
				viewer_all_valid.data().set_mesh(Vbox, Fbox);
				viewer_all_valid.data().show_faces = false;
				viewer_all_valid_gray.data().set_mesh(Vbox, Fbox);
				viewer_all_valid_gray.data().show_faces = false;
				viewer_free_boundary.data().set_mesh(Vbox, Fbox);
				viewer_free_boundary.data().show_faces = false;
				viewer_interactive.data().set_mesh(Vbox, Fbox);
				viewer_interactive.data().show_faces = false;
				viewer_t12_inspect.data().set_mesh(Vbox, Fbox);
				viewer_t12_inspect.data().show_faces = false;
			}
			std::cout << "=================================\n";
		}

		// =========================
		// VIEWER 15: EMA L1 -> L2 PROJECTION - DATA
		// =========================
		{
			// Draw L2 contours in black
			for (const auto& contour : levels[startIdx].contours) {
				int n = static_cast<int>(contour.points.size());
				if (n == 0) continue;
				Eigen::MatrixXd V(n, 3);
				for (int j = 0; j < n; ++j)
					V.row(j) = contour.points[j];
				for (int j = 0; j < n; ++j)
					viewer_proj_l1_on_l2.data().add_edges(
						V.row(j), V.row((j + 1) % n),
						Eigen::RowVector3d(0, 0, 0));
			}
			// EMA edges with at least one vertex in solid region: green
			for (const auto& e : proj_ema_l1_on_l2.edges_with_solid)
				viewer_proj_l1_on_l2.data().add_edges(
					Eigen::RowVector3d(e.first.x, e.first.y, z_l2),
					Eigen::RowVector3d(e.second.x, e.second.y, z_l2),
					Eigen::RowVector3d(0, 0.8, 0));
			// EMA edges with no vertex in solid region: red
			for (const auto& e : proj_ema_l1_on_l2.edges_without_solid)
				viewer_proj_l1_on_l2.data().add_edges(
					Eigen::RowVector3d(e.first.x, e.first.y, z_l2),
					Eigen::RowVector3d(e.second.x, e.second.y, z_l2),
					Eigen::RowVector3d(0.8, 0, 0));
			viewer_proj_l1_on_l2.data().set_mesh(Vbox, Fbox);
			viewer_proj_l1_on_l2.data().show_faces = false;
			viewer_proj_l1_on_l2.data().point_size = 2.0;
		}

		// =========================
		// VIEWER 16: EMA L2 -> L1 PROJECTION - DATA
		// =========================
		{
			// Draw L1 contours in black
			for (const auto& contour : levels[startIdx - 1].contours) {
				int n = static_cast<int>(contour.points.size());
				if (n == 0) continue;
				Eigen::MatrixXd V(n, 3);
				for (int j = 0; j < n; ++j)
					V.row(j) = contour.points[j];
				for (int j = 0; j < n; ++j)
					viewer_proj_l2_on_l1.data().add_edges(
						V.row(j), V.row((j + 1) % n),
						Eigen::RowVector3d(0, 0, 0));
			}
			// EMA edges with at least one vertex in solid region: green
			for (const auto& e : proj_ema_l2_on_l1.edges_with_solid)
				viewer_proj_l2_on_l1.data().add_edges(
					Eigen::RowVector3d(e.first.x, e.first.y, z_l1),
					Eigen::RowVector3d(e.second.x, e.second.y, z_l1),
					Eigen::RowVector3d(0, 0.8, 0));
			// EMA edges with no vertex in solid region: red
			for (const auto& e : proj_ema_l2_on_l1.edges_without_solid)
				viewer_proj_l2_on_l1.data().add_edges(
					Eigen::RowVector3d(e.first.x, e.first.y, z_l1),
					Eigen::RowVector3d(e.second.x, e.second.y, z_l1),
					Eigen::RowVector3d(0.8, 0, 0));
			viewer_proj_l2_on_l1.data().set_mesh(Vbox, Fbox);
			viewer_proj_l2_on_l1.data().show_faces = false;
			viewer_proj_l2_on_l1.data().point_size = 2.0;
		}

		// =========================
		// VORONOI L1 - CALLBACK
		// =========================
		menu_vor_l1.callback_draw_viewer_window = [&]() {
			ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(1.0f, 1.0f, 1.0f, 1.0f));
			ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.0f, 0.0f, 0.0f, 1.0f));
			ImGui::PushStyleColor(ImGuiCol_TitleBg, ImVec4(0.85f, 0.85f, 0.85f, 1.0f));
			ImGui::PushStyleColor(ImGuiCol_TitleBgActive, ImVec4(0.75f, 0.75f, 0.75f, 1.0f));

			ImGui::Begin("Voronoi L1");
			ImGui::SetWindowFontScale(1.4f);
			ImGui::Text("Contours: Black");
			ImGui::Text("Voronoi edges: Blue");
			ImGui::Text("Resample factor: %d", resampleFactor);
			ImGui::SetWindowFontScale(1.0f);
			ImGui::End();

			ImGui::PopStyleColor(4);

			ImGui::SetWindowFontScale(1.8f);
			for (const auto& lbl : labels) {
				auto screen = projectToScreen(lbl.position, viewer_vor_l1);
				ImGui::GetForegroundDrawList()->AddText(
					ImVec2(screen.x(), screen.y()),
					IM_COL32(0, 0, 0, 255),
					lbl.text.c_str());
			}
			ImGui::SetWindowFontScale(1.0f);
		};

		// =========================
		// VORONOI L2 - CALLBACK
		// =========================
		menu_vor_l2.callback_draw_viewer_window = [&]() {
			ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(1.0f, 1.0f, 1.0f, 1.0f));
			ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.0f, 0.0f, 0.0f, 1.0f));
			ImGui::PushStyleColor(ImGuiCol_TitleBg, ImVec4(0.85f, 0.85f, 0.85f, 1.0f));
			ImGui::PushStyleColor(ImGuiCol_TitleBgActive, ImVec4(0.75f, 0.75f, 0.75f, 1.0f));

			ImGui::Begin("Voronoi L2");
			ImGui::SetWindowFontScale(1.4f);
			ImGui::Text("Contours: Black");
			ImGui::Text("Voronoi edges: Blue");
			ImGui::Text("Resample factor: %d", resampleFactor);
			ImGui::SetWindowFontScale(1.0f);
			ImGui::End();

			ImGui::PopStyleColor(4);

			ImGui::SetWindowFontScale(1.8f);
			for (const auto& lbl : labels_l2) {
				auto screen = projectToScreen(lbl.position, viewer_vor_l2);
				ImGui::GetForegroundDrawList()->AddText(
					ImVec2(screen.x(), screen.y()),
					IM_COL32(0, 0, 0, 255),
					lbl.text.c_str());
			}
			ImGui::SetWindowFontScale(1.0f);
		};

		// =========================
		// IMA L1 - CALLBACK
		// =========================
		menu_ima_l1.callback_draw_viewer_window = [&]() {
			ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(1.0f, 1.0f, 1.0f, 1.0f));
			ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.0f, 0.0f, 0.0f, 1.0f));
			ImGui::PushStyleColor(ImGuiCol_TitleBg, ImVec4(0.85f, 0.85f, 0.85f, 1.0f));
			ImGui::PushStyleColor(ImGuiCol_TitleBgActive, ImVec4(0.75f, 0.75f, 0.75f, 1.0f));

			ImGui::Begin("Internal Medial Axis - L1");
			ImGui::SetWindowFontScale(1.4f);
			ImGui::TextColored(ImVec4(0.0f, 0.0f, 1.0f, 1.0f), "IMA edges: Blue");
			ImGui::Text("Contours: Black");
			ImGui::Text("IMA edges: %d", ima_l1_count);
			ImGui::Text("Discarded: %d", disc_l1_count);
			ImGui::Text("Resample: %d", resampleFactor);
			ImGui::SetWindowFontScale(1.0f);
			ImGui::End();

			ImGui::PopStyleColor(4);

			ImGui::SetWindowFontScale(1.8f);
			for (const auto& lbl : labels) {
				auto screen = projectToScreen(lbl.position, viewer_ima_l1);
				ImGui::GetForegroundDrawList()->AddText(
					ImVec2(screen.x(), screen.y()),
					IM_COL32(0, 0, 0, 255),
					lbl.text.c_str());
			}
			ImGui::SetWindowFontScale(1.0f);
		};

		// =========================
		// EMA L1 - CALLBACK
		// =========================
		menu_ema_l1.callback_draw_viewer_window = [&]() {
			ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(1.0f, 1.0f, 1.0f, 1.0f));
			ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.0f, 0.0f, 0.0f, 1.0f));
			ImGui::PushStyleColor(ImGuiCol_TitleBg, ImVec4(0.85f, 0.85f, 0.85f, 1.0f));
			ImGui::PushStyleColor(ImGuiCol_TitleBgActive, ImVec4(0.75f, 0.75f, 0.75f, 1.0f));

			ImGui::Begin("External Medial Axis - L1");
			ImGui::SetWindowFontScale(1.4f);
			ImGui::TextColored(ImVec4(0.8f, 0.0f, 0.0f, 1.0f), "EMA edges: Red");
			ImGui::Text("Contours: Black");
			ImGui::Text("EMA edges: %d", ema_l1_count);
			ImGui::Text("Discarded: %d", disc_l1_count);
			ImGui::Text("Resample: %d", resampleFactor);
			ImGui::SetWindowFontScale(1.0f);
			ImGui::End();

			ImGui::PopStyleColor(4);

			ImGui::SetWindowFontScale(1.8f);
			for (const auto& lbl : labels) {
				auto screen = projectToScreen(lbl.position, viewer_ema_l1);
				ImGui::GetForegroundDrawList()->AddText(
					ImVec2(screen.x(), screen.y()),
					IM_COL32(0, 0, 0, 255),
					lbl.text.c_str());
			}
			ImGui::SetWindowFontScale(1.0f);
		};

		// =========================
		// IMA L2 - CALLBACK
		// =========================
		menu_ima_l2.callback_draw_viewer_window = [&]() {
			ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(1.0f, 1.0f, 1.0f, 1.0f));
			ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.0f, 0.0f, 0.0f, 1.0f));
			ImGui::PushStyleColor(ImGuiCol_TitleBg, ImVec4(0.85f, 0.85f, 0.85f, 1.0f));
			ImGui::PushStyleColor(ImGuiCol_TitleBgActive, ImVec4(0.75f, 0.75f, 0.75f, 1.0f));

			ImGui::Begin("Internal Medial Axis - L2");
			ImGui::SetWindowFontScale(1.4f);
			ImGui::TextColored(ImVec4(0.0f, 0.0f, 1.0f, 1.0f), "IMA edges: Blue");
			ImGui::Text("Contours: Black");
			ImGui::Text("IMA edges: %d", ima_l2_count);
			ImGui::Text("Discarded: %d", disc_l2_count);
			ImGui::Text("Resample: %d", resampleFactor);
			ImGui::SetWindowFontScale(1.0f);
			ImGui::End();

			ImGui::PopStyleColor(4);

			ImGui::SetWindowFontScale(1.8f);
			for (const auto& lbl : labels_l2) {
				auto screen = projectToScreen(lbl.position, viewer_ima_l2);
				ImGui::GetForegroundDrawList()->AddText(
					ImVec2(screen.x(), screen.y()),
					IM_COL32(0, 0, 0, 255),
					lbl.text.c_str());
			}
			ImGui::SetWindowFontScale(1.0f);
		};

		// =========================
		// EMA L2 - CALLBACK
		// =========================
		menu_ema_l2.callback_draw_viewer_window = [&]() {
			ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(1.0f, 1.0f, 1.0f, 1.0f));
			ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.0f, 0.0f, 0.0f, 1.0f));
			ImGui::PushStyleColor(ImGuiCol_TitleBg, ImVec4(0.85f, 0.85f, 0.85f, 1.0f));
			ImGui::PushStyleColor(ImGuiCol_TitleBgActive, ImVec4(0.75f, 0.75f, 0.75f, 1.0f));

			ImGui::Begin("External Medial Axis - L2");
			ImGui::SetWindowFontScale(1.4f);
			ImGui::TextColored(ImVec4(0.8f, 0.0f, 0.0f, 1.0f), "EMA edges: Red");
			ImGui::Text("Contours: Black");
			ImGui::Text("EMA edges: %d", ema_l2_count);
			ImGui::Text("Discarded: %d", disc_l2_count);
			ImGui::Text("Resample: %d", resampleFactor);
			ImGui::SetWindowFontScale(1.0f);
			ImGui::End();

			ImGui::PopStyleColor(4);

			ImGui::SetWindowFontScale(1.8f);
			for (const auto& lbl : labels_l2) {
				auto screen = projectToScreen(lbl.position, viewer_ema_l2);
				ImGui::GetForegroundDrawList()->AddText(
					ImVec2(screen.x(), screen.y()),
					IM_COL32(0, 0, 0, 255),
					lbl.text.c_str());
			}
			ImGui::SetWindowFontScale(1.0f);
		};

		// =========================
		// IMA+EMA L1 - CALLBACK
		// =========================
		menu_both_l1.callback_draw_viewer_window = [&]() {
			ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(1.0f, 1.0f, 1.0f, 1.0f));
			ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.0f, 0.0f, 0.0f, 1.0f));
			ImGui::PushStyleColor(ImGuiCol_TitleBg, ImVec4(0.85f, 0.85f, 0.85f, 1.0f));
			ImGui::PushStyleColor(ImGuiCol_TitleBgActive, ImVec4(0.75f, 0.75f, 0.75f, 1.0f));

			ImGui::Begin("IMA + EMA - L1");
			ImGui::SetWindowFontScale(1.4f);
			ImGui::TextColored(ImVec4(0.0f, 0.0f, 1.0f, 1.0f), "IMA edges: Blue");
			ImGui::TextColored(ImVec4(0.8f, 0.0f, 0.0f, 1.0f), "EMA edges: Red");
			ImGui::Text("Contours: Black");
			ImGui::Separator();
			ImGui::Text("IMA: %d  EMA: %d", ima_l1_count, ema_l1_count);
			ImGui::Text("Discarded: %d", disc_l1_count);
			ImGui::Text("Resample: %d", resampleFactor);
			ImGui::SetWindowFontScale(1.0f);
			ImGui::End();

			ImGui::PopStyleColor(4);

			ImGui::SetWindowFontScale(1.8f);
			for (const auto& lbl : labels) {
				auto screen = projectToScreen(lbl.position, viewer_both_l1);
				ImGui::GetForegroundDrawList()->AddText(
					ImVec2(screen.x(), screen.y()),
					IM_COL32(0, 0, 0, 255),
					lbl.text.c_str());
			}
			ImGui::SetWindowFontScale(1.0f);
		};

		// =========================
		// IMA+EMA L2 - CALLBACK
		// =========================
		menu_both_l2.callback_draw_viewer_window = [&]() {
			ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(1.0f, 1.0f, 1.0f, 1.0f));
			ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.0f, 0.0f, 0.0f, 1.0f));
			ImGui::PushStyleColor(ImGuiCol_TitleBg, ImVec4(0.85f, 0.85f, 0.85f, 1.0f));
			ImGui::PushStyleColor(ImGuiCol_TitleBgActive, ImVec4(0.75f, 0.75f, 0.75f, 1.0f));

			ImGui::Begin("IMA + EMA - L2");
			ImGui::SetWindowFontScale(1.4f);
			ImGui::TextColored(ImVec4(0.0f, 0.0f, 1.0f, 1.0f), "IMA edges: Blue");
			ImGui::TextColored(ImVec4(0.8f, 0.0f, 0.0f, 1.0f), "EMA edges: Red");
			ImGui::Text("Contours: Black");
			ImGui::Separator();
			ImGui::Text("IMA: %d  EMA: %d", ima_l2_count, ema_l2_count);
			ImGui::Text("Discarded: %d", disc_l2_count);
			ImGui::Text("Resample: %d", resampleFactor);
			ImGui::SetWindowFontScale(1.0f);
			ImGui::End();

			ImGui::PopStyleColor(4);

			ImGui::SetWindowFontScale(1.8f);
			for (const auto& lbl : labels_l2) {
				auto screen = projectToScreen(lbl.position, viewer_both_l2);
				ImGui::GetForegroundDrawList()->AddText(
					ImVec2(screen.x(), screen.y()),
					IM_COL32(0, 0, 0, 255),
					lbl.text.c_str());
			}
			ImGui::SetWindowFontScale(1.0f);
		};
		// =========================
		// EMA L1->L2 PROJECTION - CALLBACK
		// =========================
		menu_proj_l1_on_l2.callback_draw_viewer_window = [&]() {
			ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(1.0f, 1.0f, 1.0f, 1.0f));
			ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.0f, 0.0f, 0.0f, 1.0f));
			ImGui::PushStyleColor(ImGuiCol_TitleBg, ImVec4(0.85f, 0.85f, 0.85f, 1.0f));
			ImGui::PushStyleColor(ImGuiCol_TitleBgActive, ImVec4(0.75f, 0.75f, 0.75f, 1.0f));

			ImGui::Begin("EMA L1 -> L2 Projection");
			ImGui::SetWindowFontScale(1.4f);
			ImGui::TextColored(ImVec4(0.0f, 0.8f, 0.0f, 1.0f), "In solid: Green");
			ImGui::TextColored(ImVec4(0.8f, 0.0f, 0.0f, 1.0f), "Not in solid: Red");
			ImGui::Text("L2 Contours: Black");
			ImGui::Separator();
			ImGui::Text("Edges in solid: %d  Not: %d",
				(int)proj_ema_l1_on_l2.edges_with_solid.size(),
				(int)proj_ema_l1_on_l2.edges_without_solid.size());
			ImGui::Text("Solid vertices: %d  Void: %d",
				(int)proj_ema_l1_on_l2.solid_vertices.size(),
				(int)proj_ema_l1_on_l2.void_vertices.size());
			ImGui::Separator();
			ImGui::Text("Augmented L2: %d pts", (int)augmented_points_l2.size());
			ImGui::Text("  Contour: %d", aug_contour_l2);
			ImGui::Text("  IMA: %d", aug_ima_l2);
			ImGui::Text("  EMA proj: %d", aug_ema_proj_l2);
			ImGui::SetWindowFontScale(1.0f);
			ImGui::End();

			ImGui::PopStyleColor(4);

			ImGui::SetWindowFontScale(1.8f);
			for (const auto& lbl : labels_l2) {
				auto screen = projectToScreen(lbl.position, viewer_proj_l1_on_l2);
				ImGui::GetForegroundDrawList()->AddText(
					ImVec2(screen.x(), screen.y()),
					IM_COL32(0, 0, 0, 255),
					lbl.text.c_str());
			}
			ImGui::SetWindowFontScale(1.0f);
		};

		// =========================
		// EMA L2->L1 PROJECTION - CALLBACK
		// =========================
		menu_proj_l2_on_l1.callback_draw_viewer_window = [&]() {
			ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(1.0f, 1.0f, 1.0f, 1.0f));
			ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.0f, 0.0f, 0.0f, 1.0f));
			ImGui::PushStyleColor(ImGuiCol_TitleBg, ImVec4(0.85f, 0.85f, 0.85f, 1.0f));
			ImGui::PushStyleColor(ImGuiCol_TitleBgActive, ImVec4(0.75f, 0.75f, 0.75f, 1.0f));

			ImGui::Begin("EMA L2 -> L1 Projection");
			ImGui::SetWindowFontScale(1.4f);
			ImGui::TextColored(ImVec4(0.0f, 0.8f, 0.0f, 1.0f), "In solid: Green");
			ImGui::TextColored(ImVec4(0.8f, 0.0f, 0.0f, 1.0f), "Not in solid: Red");
			ImGui::Text("L1 Contours: Black");
			ImGui::Separator();
			ImGui::Text("Edges in solid: %d  Not: %d",
				(int)proj_ema_l2_on_l1.edges_with_solid.size(),
				(int)proj_ema_l2_on_l1.edges_without_solid.size());
			ImGui::Text("Solid vertices: %d  Void: %d",
				(int)proj_ema_l2_on_l1.solid_vertices.size(),
				(int)proj_ema_l2_on_l1.void_vertices.size());
			ImGui::Separator();
			ImGui::Text("Augmented L1: %d pts", (int)augmented_points_l1.size());
			ImGui::Text("  Contour: %d", aug_contour_l1);
			ImGui::Text("  IMA: %d", aug_ima_l1);
			ImGui::Text("  EMA proj: %d", aug_ema_proj_l1);
			ImGui::SetWindowFontScale(1.0f);
			ImGui::End();

			ImGui::PopStyleColor(4);

			ImGui::SetWindowFontScale(1.8f);
			for (const auto& lbl : labels) {
				auto screen = projectToScreen(lbl.position, viewer_proj_l2_on_l1);
				ImGui::GetForegroundDrawList()->AddText(
					ImVec2(screen.x(), screen.y()),
					IM_COL32(0, 0, 0, 255),
					lbl.text.c_str());
			}
			ImGui::SetWindowFontScale(1.0f);
		};
		} // end algo 1/2/3/4 single-Delaunay block

		// =========================
		// 3D DELAUNAY - CALLBACK
		// =========================
		menu_delaunay3d.callback_draw_viewer_window = [&]() {
			ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(1.0f, 1.0f, 1.0f, 1.0f));
			ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.0f, 0.0f, 0.0f, 1.0f));
			ImGui::PushStyleColor(ImGuiCol_TitleBg, ImVec4(0.85f, 0.85f, 0.85f, 1.0f));
			ImGui::PushStyleColor(ImGuiCol_TitleBgActive, ImVec4(0.75f, 0.75f, 0.75f, 1.0f));

			ImGui::Begin("3D Delaunay Triangulation");
			ImGui::SetWindowFontScale(1.4f);
			ImGui::Text("Points: %d", delaunay3d_num_points);
			ImGui::Text("  L1: %d", (int)augmented_points_l1.size());
			ImGui::Text("  L2: %d", (int)augmented_points_l2.size());
			ImGui::Separator();
			ImGui::Text("Tetrahedra: %d", delaunay3d_num_tets);
			ImGui::Text("Boundary faces: %d", delaunay3d_num_faces);
			ImGui::Separator();
			ImGui::Text("Classification:");
			ImGui::Text("  T1  (3L1+1L2): %d", delaunay3d_T1);
			ImGui::Text("  T2  (1L1+3L2): %d", delaunay3d_T2);
			ImGui::Text("  T12 (2L1+2L2): %d", delaunay3d_T12);
			if (delaunay3d_degenerate > 0)
				ImGui::Text("  Degenerate:    %d", delaunay3d_degenerate);
			ImGui::Separator();
			ImGui::Text("Resample: %d", resampleFactor);
			ImGui::SetWindowFontScale(1.0f);
			ImGui::End();

			ImGui::PopStyleColor(4);
		};

		// =========================
		// T1 CALLBACK
		// =========================
		menu_t1.callback_draw_viewer_window = [&]() {
			ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(1.0f, 1.0f, 1.0f, 1.0f));
			ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.0f, 0.0f, 0.0f, 1.0f));
			ImGui::PushStyleColor(ImGuiCol_TitleBg, ImVec4(0.85f, 0.85f, 0.85f, 1.0f));
			ImGui::PushStyleColor(ImGuiCol_TitleBgActive, ImVec4(0.75f, 0.75f, 0.75f, 1.0f));

			ImGui::Begin("T1 - Base L1, Apex L2");
			ImGui::SetWindowFontScale(1.4f);
			ImGui::Text("3 vertices in L1 + 1 vertex in L2");
			ImGui::Separator();
			ImGui::Text("Tetrahedra: %d", delaunay3d_T1);
			ImGui::Text("Faces: %d", delaunay3d_T1_faces);
			ImGui::SetWindowFontScale(1.0f);
			ImGui::End();

			ImGui::PopStyleColor(4);
		};

		// =========================
		// T2 CALLBACK
		// =========================
		menu_t2.callback_draw_viewer_window = [&]() {
			ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(1.0f, 1.0f, 1.0f, 1.0f));
			ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.0f, 0.0f, 0.0f, 1.0f));
			ImGui::PushStyleColor(ImGuiCol_TitleBg, ImVec4(0.85f, 0.85f, 0.85f, 1.0f));
			ImGui::PushStyleColor(ImGuiCol_TitleBgActive, ImVec4(0.75f, 0.75f, 0.75f, 1.0f));

			ImGui::Begin("T2 - Base L2, Apex L1");
			ImGui::SetWindowFontScale(1.4f);
			ImGui::Text("1 vertex in L1 + 3 vertices in L2");
			ImGui::Separator();
			ImGui::Text("Tetrahedra: %d", delaunay3d_T2);
			ImGui::Text("Faces: %d", delaunay3d_T2_faces);
			ImGui::SetWindowFontScale(1.0f);
			ImGui::End();

			ImGui::PopStyleColor(4);
		};

		// =========================
		// T12 CALLBACK
		// =========================
		menu_t12.callback_draw_viewer_window = [&]() {
			ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(1.0f, 1.0f, 1.0f, 1.0f));
			ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.0f, 0.0f, 0.0f, 1.0f));
			ImGui::PushStyleColor(ImGuiCol_TitleBg, ImVec4(0.85f, 0.85f, 0.85f, 1.0f));
			ImGui::PushStyleColor(ImGuiCol_TitleBgActive, ImVec4(0.75f, 0.75f, 0.75f, 1.0f));

			ImGui::Begin("T12 - Edge L1, Edge L2");
			ImGui::SetWindowFontScale(1.4f);
			ImGui::Text("2 vertices in L1 + 2 vertices in L2");
			ImGui::Separator();
			ImGui::Text("Tetrahedra: %d", delaunay3d_T12);
			ImGui::Text("Faces: %d", delaunay3d_T12_faces);
			ImGui::SetWindowFontScale(1.0f);
			ImGui::End();

			ImGui::PopStyleColor(4);
		};

		// =========================
		// T1 VALID CALLBACK
		// =========================
		menu_t1_valid.callback_draw_viewer_window = [&]() {
			ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(1.0f, 1.0f, 1.0f, 1.0f));
			ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.0f, 0.0f, 0.0f, 1.0f));
			ImGui::PushStyleColor(ImGuiCol_TitleBg, ImVec4(0.85f, 0.85f, 0.85f, 1.0f));
			ImGui::PushStyleColor(ImGuiCol_TitleBgActive, ImVec4(0.75f, 0.75f, 0.75f, 1.0f));

			ImGui::Begin("T1 Valid - Base in Solid L1");
			ImGui::SetWindowFontScale(1.4f);
			ImGui::Text("T1 with base triangle inside solid L1");
			ImGui::Separator();
			ImGui::Text("Valid:   %d tets, %d faces", delaunay3d_T1_valid, delaunay3d_T1_valid_faces);
			ImGui::Text("Removed: %d tets, %d faces", delaunay3d_T1_removed, delaunay3d_T1_removed_faces);
			if (global_t1_removed_by_midpoint > 0)
				ImGui::Text("  Edge-midpoint: %d", global_t1_removed_by_midpoint);
			ImGui::Text("Total T1: %d", delaunay3d_T1);
			ImGui::SetWindowFontScale(1.0f);
			ImGui::End();

			ImGui::PopStyleColor(4);
		};

		// =========================
		// T1 REMOVED CALLBACK
		// =========================
		menu_t1_removed.callback_draw_viewer_window = [&]() {
			ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(1.0f, 1.0f, 1.0f, 1.0f));
			ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.0f, 0.0f, 0.0f, 1.0f));
			ImGui::PushStyleColor(ImGuiCol_TitleBg, ImVec4(0.85f, 0.85f, 0.85f, 1.0f));
			ImGui::PushStyleColor(ImGuiCol_TitleBgActive, ImVec4(0.75f, 0.75f, 0.75f, 1.0f));

			ImGui::Begin("T1 Removed - Base outside Solid L1");
			ImGui::SetWindowFontScale(1.4f);
			ImGui::Text("T1 with base triangle outside solid L1");
			ImGui::Separator();
			ImGui::Text("Removed: %d tets, %d faces", delaunay3d_T1_removed, delaunay3d_T1_removed_faces);
			ImGui::Text("Valid:   %d tets, %d faces", delaunay3d_T1_valid, delaunay3d_T1_valid_faces);
			ImGui::Text("Total T1: %d", delaunay3d_T1);
			ImGui::SetWindowFontScale(1.0f);
			ImGui::End();

			ImGui::PopStyleColor(4);
		};

		// =========================
		// T2 VALID CALLBACK
		// =========================
		menu_t2_valid.callback_draw_viewer_window = [&]() {
			ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(1.0f, 1.0f, 1.0f, 1.0f));
			ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.0f, 0.0f, 0.0f, 1.0f));
			ImGui::PushStyleColor(ImGuiCol_TitleBg, ImVec4(0.85f, 0.85f, 0.85f, 1.0f));
			ImGui::PushStyleColor(ImGuiCol_TitleBgActive, ImVec4(0.75f, 0.75f, 0.75f, 1.0f));

			ImGui::Begin("T2 Valid - Base in Solid L2");
			ImGui::SetWindowFontScale(1.4f);
			ImGui::Text("T2 with base triangle inside solid L2");
			ImGui::Separator();
			ImGui::Text("Valid:   %d tets, %d faces", delaunay3d_T2_valid, delaunay3d_T2_valid_faces);
			ImGui::Text("Removed: %d tets, %d faces", delaunay3d_T2_removed, delaunay3d_T2_removed_faces);
			if (global_t2_removed_by_midpoint > 0)
				ImGui::Text("  Edge-midpoint: %d", global_t2_removed_by_midpoint);
			ImGui::Text("Total T2: %d", delaunay3d_T2);
			ImGui::SetWindowFontScale(1.0f);
			ImGui::End();

			ImGui::PopStyleColor(4);
		};

		// =========================
		// T2 REMOVED CALLBACK
		// =========================
		menu_t2_removed.callback_draw_viewer_window = [&]() {
			ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(1.0f, 1.0f, 1.0f, 1.0f));
			ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.0f, 0.0f, 0.0f, 1.0f));
			ImGui::PushStyleColor(ImGuiCol_TitleBg, ImVec4(0.85f, 0.85f, 0.85f, 1.0f));
			ImGui::PushStyleColor(ImGuiCol_TitleBgActive, ImVec4(0.75f, 0.75f, 0.75f, 1.0f));

			ImGui::Begin("T2 Removed - Base outside Solid L2");
			ImGui::SetWindowFontScale(1.4f);
			ImGui::Text("T2 with base triangle outside solid L2");
			ImGui::Separator();
			ImGui::Text("Removed: %d tets, %d faces", delaunay3d_T2_removed, delaunay3d_T2_removed_faces);
			ImGui::Text("Valid:   %d tets, %d faces", delaunay3d_T2_valid, delaunay3d_T2_valid_faces);
			ImGui::Text("Total T2: %d", delaunay3d_T2);
			ImGui::SetWindowFontScale(1.0f);
			ImGui::End();

			ImGui::PopStyleColor(4);
		};

		// =========================
		// T12 VALID CALLBACK
		// =========================
		menu_t12_valid.callback_draw_viewer_window = [&]() {
			ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(1.0f, 1.0f, 1.0f, 1.0f));
			ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.0f, 0.0f, 0.0f, 1.0f));
			ImGui::PushStyleColor(ImGuiCol_TitleBg, ImVec4(0.85f, 0.85f, 0.85f, 1.0f));
			ImGui::PushStyleColor(ImGuiCol_TitleBgActive, ImVec4(0.75f, 0.75f, 0.75f, 1.0f));

			ImGui::Begin("T12 Valid");
			ImGui::SetWindowFontScale(1.4f);
			ImGui::Text("T12 passing midpoint + isolated filters");
			ImGui::Separator();
			ImGui::Text("Valid:   %d tets, %d faces", delaunay3d_T12_valid, delaunay3d_T12_valid_faces);
			ImGui::Text("Removed: %d tets, %d faces", delaunay3d_T12_removed, delaunay3d_T12_removed_faces);
			ImGui::Text("Total T12: %d", delaunay3d_T12);
			ImGui::SetWindowFontScale(1.0f);
			ImGui::End();

			ImGui::PopStyleColor(4);
		};

		// =========================
		// T12 REMOVED CALLBACK
		// =========================
		menu_t12_removed.callback_draw_viewer_window = [&]() {
			ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(1.0f, 1.0f, 1.0f, 1.0f));
			ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.0f, 0.0f, 0.0f, 1.0f));
			ImGui::PushStyleColor(ImGuiCol_TitleBg, ImVec4(0.85f, 0.85f, 0.85f, 1.0f));
			ImGui::PushStyleColor(ImGuiCol_TitleBgActive, ImVec4(0.75f, 0.75f, 0.75f, 1.0f));

			ImGui::Begin("T12 Removed");
			ImGui::SetWindowFontScale(1.4f);
			ImGui::Text("T12 failing midpoint or isolated filters");
			ImGui::Separator();
			ImGui::Text("Removed: %d tets, %d faces", delaunay3d_T12_removed, delaunay3d_T12_removed_faces);
			ImGui::Text("Valid:   %d tets, %d faces", delaunay3d_T12_valid, delaunay3d_T12_valid_faces);
			ImGui::Text("Total T12: %d", delaunay3d_T12);
			ImGui::SetWindowFontScale(1.0f);
			ImGui::End();

			ImGui::PopStyleColor(4);
		};

		// =========================
		// ALL VALID CALLBACK
		// =========================
		menu_all_valid.callback_draw_viewer_window = [&]() {
			ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(1.0f, 1.0f, 1.0f, 1.0f));
			ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.0f, 0.0f, 0.0f, 1.0f));
			ImGui::PushStyleColor(ImGuiCol_TitleBg, ImVec4(0.85f, 0.85f, 0.85f, 1.0f));
			ImGui::PushStyleColor(ImGuiCol_TitleBgActive, ImVec4(0.75f, 0.75f, 0.75f, 1.0f));

			ImGui::Begin("All Valid Tetrahedra");
			ImGui::SetWindowFontScale(1.4f);
			ImGui::Text("T1 valid + T2 valid + T12 valid");
			ImGui::Separator();
			ImGui::TextColored(ImVec4(0.4f, 0.6f, 1.0f, 1.0f), "T1 valid:  %d tets, %d faces", delaunay3d_T1_valid, delaunay3d_T1_valid_faces);
			ImGui::TextColored(ImVec4(0.4f, 0.6f, 1.0f, 1.0f), "T2 valid:  %d tets, %d faces", delaunay3d_T2_valid, delaunay3d_T2_valid_faces);
			ImGui::TextColored(ImVec4(0.4f, 0.6f, 1.0f, 1.0f), "T12 valid: %d tets, %d faces", delaunay3d_T12_valid, delaunay3d_T12_valid_faces);
			ImGui::Separator();
			ImGui::Text("Total valid faces: %d", delaunay3d_all_valid_faces);
			ImGui::Separator();
			ImGui::Text("Removed:");
			ImGui::Text("  T1:  %d tets", delaunay3d_T1_removed);
			ImGui::Text("  T2:  %d tets", delaunay3d_T2_removed);
			ImGui::Text("  T12: %d tets", delaunay3d_T12_removed);
			ImGui::SetWindowFontScale(1.0f);
			ImGui::End();

			ImGui::PopStyleColor(4);
		};

		// =========================
		// ALL VALID GRAY CALLBACK
		// =========================
		menu_all_valid_gray.callback_draw_viewer_window = [&]() {
			ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(1.0f, 1.0f, 1.0f, 1.0f));
			ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.0f, 0.0f, 0.0f, 1.0f));
			ImGui::PushStyleColor(ImGuiCol_TitleBg, ImVec4(0.85f, 0.85f, 0.85f, 1.0f));
			ImGui::PushStyleColor(ImGuiCol_TitleBgActive, ImVec4(0.75f, 0.75f, 0.75f, 1.0f));

			ImGui::Begin("All Valid Tetrahedra (Gray)");
			ImGui::SetWindowFontScale(1.4f);
			ImGui::Text("T1 valid + T2 valid + T12 valid (uniform gray)");
			ImGui::Separator();
			ImGui::Text("T1 valid:  %d tets, %d faces", delaunay3d_T1_valid, delaunay3d_T1_valid_faces);
			ImGui::Text("T2 valid:  %d tets, %d faces", delaunay3d_T2_valid, delaunay3d_T2_valid_faces);
			ImGui::Text("T12 valid: %d tets, %d faces", delaunay3d_T12_valid, delaunay3d_T12_valid_faces);
			ImGui::Separator();
			ImGui::Text("Total valid faces: %d", delaunay3d_all_valid_faces);
			ImGui::SetWindowFontScale(1.0f);
			ImGui::End();

			ImGui::PopStyleColor(4);
		};

		// =========================
		// FREE BOUNDARY CALLBACK
		// =========================
		menu_free_boundary.callback_draw_viewer_window = [&]() {
			ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(1.0f, 1.0f, 1.0f, 1.0f));
			ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.0f, 0.0f, 0.0f, 1.0f));
			ImGui::PushStyleColor(ImGuiCol_TitleBg, ImVec4(0.85f, 0.85f, 0.85f, 1.0f));
			ImGui::PushStyleColor(ImGuiCol_TitleBgActive, ImVec4(0.75f, 0.75f, 0.75f, 1.0f));

			ImGui::Begin("Free Boundary (Skin)");
			ImGui::SetWindowFontScale(1.4f);
			ImGui::Text("Surface of all valid tetrahedra");
			ImGui::Text("Faces appearing exactly once = exterior skin");
			ImGui::Separator();
			ImGui::Text("Total tet faces (all valid): %d", delaunay3d_all_valid_faces);
			ImGui::Text("Free boundary faces: %d", delaunay3d_free_boundary_faces);
			ImGui::Text("Interior faces (shared): %d", (delaunay3d_all_valid_faces - delaunay3d_free_boundary_faces) / 2);
			ImGui::Separator();
			ImGui::Text("=== Manifold Diagnostic ===");
			if (global_manifold_info.is_manifold())
				ImGui::TextColored(ImVec4(0.0f, 0.6f, 0.0f, 1.0f), "Result: MANIFOLD");
			else
				ImGui::TextColored(ImVec4(0.8f, 0.0f, 0.0f, 1.0f), "Result: NOT MANIFOLD");
			ImGui::Text("Edge-manifold: %s", global_manifold_info.edge_manifold ? "YES" : "NO");
			ImGui::Text("Vertex-manifold: %s", global_manifold_info.vertex_manifold ? "YES" : "NO");
			ImGui::Text("Euler characteristic: %d", global_manifold_info.euler_characteristic);
			if (!global_manifold_info.vertex_manifold)
				ImGui::TextColored(ImVec4(0.8f, 0.0f, 0.0f, 1.0f), "Non-manifold vertices: %d", global_manifold_info.non_manifold_vertices);
			if (!global_manifold_info.edge_manifold)
				ImGui::TextColored(ImVec4(0.8f, 0.0f, 0.0f, 1.0f), "Non-manifold edges: %d", global_manifold_info.non_manifold_edges);
			ImGui::SetWindowFontScale(1.0f);
			ImGui::End();

			ImGui::PopStyleColor(4);
		};

		// =========================
		// T12 INSPECT CALLBACK
		// =========================
		menu_t12_inspect.callback_draw_viewer_window = [&]() {
			ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(1.0f, 1.0f, 1.0f, 1.0f));
			ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.0f, 0.0f, 0.0f, 1.0f));
			ImGui::PushStyleColor(ImGuiCol_TitleBg, ImVec4(0.85f, 0.85f, 0.85f, 1.0f));
			ImGui::PushStyleColor(ImGuiCol_TitleBgActive, ImVec4(0.75f, 0.75f, 0.75f, 1.0f));

			ImGui::Begin("Tet Inspector");
			ImGui::SetWindowFontScale(1.4f);

		if (global_t1t2_inspect_idx >= 0) {
				const char* type_name = (global_t1t2_inspect_type == 1) ? "T1" : "T2";
				if (global_t1t2_inspect_is_removed)
					ImGui::TextColored(ImVec4(0.8f, 0.2f, 0.2f, 1.0f), "Selected %s removed #%d", type_name, global_t1t2_inspect_idx);
				else
					ImGui::Text("Selected %s valid #%d", type_name, global_t1t2_inspect_idx);

				const auto& tets_list = (global_t1t2_inspect_type == 1) ? global_t1_tets : global_t2_tets;
				if (global_t1t2_inspect_idx < (int)tets_list.size()) {
					const auto& tet = tets_list[global_t1t2_inspect_idx];
					ImGui::Text("Centroid: (%.4f, %.4f) -> %s", tet.centroid_x, tet.centroid_y,
						tet.centroid_in_solid ? "SOLID" : "VOID");
					ImGui::Separator();
					ImGui::Text("Base verts: [%d, %d, %d]", (int)tet.base_verts[0], (int)tet.base_verts[1], (int)tet.base_verts[2]);
					ImGui::Text("Apex vert: %d", (int)tet.apex_vert);
					ImGui::Separator();
					ImGui::Text("Edge samples (base->apex, t=2/5,1/2,3/5):");
					const char* sample_names[3] = {"t=2/5", "t=1/2", "t=3/5"};
					for (int be = 0; be < 3; ++be) {
						ImGui::Text("  Edge [%d]->apex:", (int)tet.base_verts[be]);
						for (int si = 0; si < 3; ++si) {
							const char* l1_status = tet.midpoint_in_solid_l1[be][si] ? "SOLID" : "VOID";
							const char* l2_status = tet.midpoint_in_solid_l2[be][si] ? "SOLID" : "VOID";
							ImVec4 color = (tet.midpoint_in_solid_l1[be][si] || tet.midpoint_in_solid_l2[be][si])
								? ImVec4(0.0f, 0.5f, 0.0f, 1.0f) : ImVec4(0.8f, 0.0f, 0.0f, 1.0f);
							ImGui::TextColored(color, "    %s: (%.4f, %.4f) L1=%s L2=%s",
								sample_names[si], tet.midpoint_x[be][si], tet.midpoint_y[be][si], l1_status, l2_status);
						}
					}
					ImGui::Separator();
					ImGui::Text("Void count: L1=%d/9, L2=%d/9", tet.void_l1_count, tet.void_l2_count);
					if (tet.fails_midpoint)
						ImGui::TextColored(ImVec4(0.8f, 0.0f, 0.0f, 1.0f), "Midpoint filter: FAIL (all 9 void on both levels)");
					else
						ImGui::TextColored(ImVec4(0.0f, 0.5f, 0.0f, 1.0f), "Midpoint filter: PASS");
				}
		} else if (global_t12_inspect_idx < 0) {
				ImGui::Text("No tet selected.");
				ImGui::Text("Click a face in Interactive viewer.");
			} else {
				if (global_t12_inspect_is_removed)
					ImGui::TextColored(ImVec4(0.8f, 0.2f, 0.2f, 1.0f), "Selected removed T12: #%d", global_t12_inspect_idx);
				else
					ImGui::Text("Selected valid T12: #%d", global_t12_inspect_idx);

				// Find the actual global_t12_tets index for this T12
				auto make_face_key = [](index_t a, index_t b, index_t c) -> std::array<index_t, 3> {
					std::array<index_t, 3> f = {a, b, c};
					if (f[0] > f[1]) std::swap(f[0], f[1]);
					if (f[1] > f[2]) std::swap(f[1], f[2]);
					if (f[0] > f[1]) std::swap(f[0], f[1]);
					return f;
				};

				// Choose the appropriate face list based on valid/removed
				const auto& inspect_face_list = global_t12_inspect_is_removed
					? global_faces_t12_removed : global_faces_t12_valid;

				int match_count = 0;
				int real_idx = -1;
				for (size_t i = 0; i < global_t12_tets.size(); ++i) {
					if (match_count * 4 < (int)inspect_face_list.size()) {
						auto fk_tet = make_face_key(
							global_t12_tets[i].faces[0][0],
							global_t12_tets[i].faces[0][1],
							global_t12_tets[i].faces[0][2]);
						int pos = match_count * 4;
						if (pos < (int)inspect_face_list.size()) {
							auto fk_list = make_face_key(
								inspect_face_list[pos][0],
								inspect_face_list[pos][1],
								inspect_face_list[pos][2]);
							if (fk_tet == fk_list) {
								if (match_count == global_t12_inspect_idx) {
									real_idx = static_cast<int>(i);
									break;
								}
								match_count++;
							}
						}
					}
				}

				if (real_idx >= 0) {
					const auto& tet = global_t12_tets[real_idx];

					// Build T1v + T2v face set
					std::set<std::array<index_t, 3>> t1t2_faces;
					for (const auto& f : global_faces_t1_valid)
						t1t2_faces.insert(make_face_key(f[0], f[1], f[2]));
					for (const auto& f : global_faces_t2_valid)
						t1t2_faces.insert(make_face_key(f[0], f[1], f[2]));

					// Build T12v face set EXCLUDING this tet's own faces
					// A face counts as "shared with another T12" only if it appears
					// in a DIFFERENT valid T12 tet (not this one)
					std::set<std::array<index_t, 3>> own_faces;
					for (int lf = 0; lf < 4; ++lf)
						own_faces.insert(make_face_key(tet.faces[lf][0], tet.faces[lf][1], tet.faces[lf][2]));

					// Count how many times each face appears across ALL valid T12 tets
					std::map<std::array<index_t, 3>, int> t12v_face_count;
					for (size_t fi = 0; fi < global_faces_t12_valid.size(); ++fi) {
						auto fk = make_face_key(global_faces_t12_valid[fi][0], global_faces_t12_valid[fi][1], global_faces_t12_valid[fi][2]);
						t12v_face_count[fk]++;
					}

					// Classify each face
					int shared_t1t2 = 0, shared_t12 = 0, boundary = 0;
					for (int lf = 0; lf < 4; ++lf) {
						auto fk = make_face_key(tet.faces[lf][0], tet.faces[lf][1], tet.faces[lf][2]);
						bool in_t1t2 = t1t2_faces.count(fk) > 0;
						// Shared with another T12 if the face appears more than once in valid T12 faces
						// (once for this tet, plus at least once for another)
						auto it_cnt = t12v_face_count.find(fk);
						bool shared_other_t12 = (it_cnt != t12v_face_count.end() && it_cnt->second > 1);

						const char* label = "(boundary)";
						if (in_t1t2) { shared_t1t2++; label = "-> T1/T2"; }
						else if (shared_other_t12) { shared_t12++; label = "-> T12"; }
						else { boundary++; }

						ImGui::Text("Face %d: [%d,%d,%d] %s", lf,
							(int)tet.faces[lf][0], (int)tet.faces[lf][1], (int)tet.faces[lf][2],
							label);
					}
					ImGui::Separator();
					ImGui::Text("Shared with T1/T2: %d faces", shared_t1t2);
					ImGui::Text("Shared with T12v:  %d faces", shared_t12);
					ImGui::Text("Boundary:          %d faces", boundary);
					ImGui::Separator();
					if (global_t12_inspect_is_removed) {
						ImGui::Separator();
						ImGui::TextColored(ImVec4(0.8f, 0.2f, 0.2f, 1.0f), "STATUS: REMOVED");
					}
					ImGui::Text("Midpoint: %s", tet.fails_midpoint ? "FAIL" : "pass");
					ImGui::Text("Isolated: %s", tet.fails_isolated ? "FAIL" : "pass");
					bool currently_isolated = (shared_t1t2 == 0 && shared_t12 == 0);
					if (currently_isolated)
						ImGui::TextColored(ImVec4(0.8f, 0.0f, 0.0f, 1.0f), "(no T1/T2 or T12 neighbors)");
					else if (shared_t1t2 > 0)
						ImGui::Text("(directly touches T1/T2)");
					else
						ImGui::Text("(connected via T12 chain)");
				} else {
					ImGui::TextColored(ImVec4(0.8f, 0.2f, 0.2f, 1.0f), "Could not find tet");
				}
			}

			ImGui::SetWindowFontScale(1.0f);
			ImGui::End();
			ImGui::PopStyleColor(4);

			// Rebuild inspect viewer mesh
			if (global_t12_inspect_needs_rebuild) {
				global_t12_inspect_needs_rebuild = false;
				viewer_t12_inspect.data().clear();

				if (global_t1t2_inspect_idx >= 0 && global_V3d.rows() > 0) {
					// Show the selected T1/T2 tet
					const auto& tets_list = (global_t1t2_inspect_type == 1) ? global_t1_tets : global_t2_tets;
					if (global_t1t2_inspect_idx < (int)tets_list.size()) {
						const auto& tet = tets_list[global_t1t2_inspect_idx];
						int nf = 4;
						Eigen::MatrixXi F(nf, 3);
						for (int i = 0; i < nf; ++i)
							F.row(i) << static_cast<int>(tet.faces[i][0]),
								static_cast<int>(tet.faces[i][1]),
								static_cast<int>(tet.faces[i][2]);
						viewer_t12_inspect.data().set_mesh(global_V3d, F);
						viewer_t12_inspect.data().show_lines = true;
						viewer_t12_inspect.data().show_faces = true;
						Eigen::MatrixXd C(nf, 3);
						Eigen::RowVector3d color = global_t1t2_inspect_is_removed
							? Eigen::RowVector3d(0.8, 0.2, 0.2)
							: (global_t1t2_inspect_type == 1 ? Eigen::RowVector3d(0.4, 0.6, 1.0) : Eigen::RowVector3d(0.4, 1.0, 0.4));
						for (int i = 0; i < nf; ++i)
							C.row(i) = color;
						viewer_t12_inspect.data().set_colors(C);
					}
				}
				else if (global_t12_inspect_idx >= 0 && global_V3d.rows() > 0) {
					auto make_fk = [](index_t a, index_t b, index_t c) -> std::array<index_t, 3> {
						std::array<index_t, 3> f = {a, b, c};
						if (f[0] > f[1]) std::swap(f[0], f[1]);
						if (f[1] > f[2]) std::swap(f[1], f[2]);
						if (f[0] > f[1]) std::swap(f[0], f[1]);
						return f;
					};

					// Find the real tet index (same logic as above)
					const auto& rebuild_face_list = global_t12_inspect_is_removed
						? global_faces_t12_removed : global_faces_t12_valid;
					int match_count2 = 0;
					int real_idx = -1;
					for (size_t i = 0; i < global_t12_tets.size(); ++i) {
						if (match_count2 * 4 < (int)rebuild_face_list.size()) {
							auto fk_tet = make_fk(
								global_t12_tets[i].faces[0][0],
								global_t12_tets[i].faces[0][1],
								global_t12_tets[i].faces[0][2]);
							int pos = match_count2 * 4;
							if (pos < (int)rebuild_face_list.size()) {
								auto fk_list = make_fk(
									rebuild_face_list[pos][0],
									rebuild_face_list[pos][1],
									rebuild_face_list[pos][2]);
								if (fk_tet == fk_list) {
									if (match_count2 == global_t12_inspect_idx) {
										real_idx = static_cast<int>(i);
										break;
									}
									match_count2++;
								}
							}
						}
					}

					if (real_idx >= 0) {
						const auto& tet = global_t12_tets[real_idx];

						// Build face sets for T1v and T2v separately
						std::set<std::array<index_t, 3>> t1v_set, t2v_set;
						for (const auto& f : global_faces_t1_valid)
							t1v_set.insert(make_fk(f[0], f[1], f[2]));
						for (const auto& f : global_faces_t2_valid)
							t2v_set.insert(make_fk(f[0], f[1], f[2]));

						// Collect faces: the 4 T12 faces + matching T1/T2 neighbor tets
						std::vector<std::array<index_t, 3>> inspect_faces;
						std::vector<int> inspect_types; // 12=T12, 1=T1, 2=T2

						// Add the 4 faces of the selected T12
						for (int lf = 0; lf < 4; ++lf) {
							inspect_faces.push_back(tet.faces[lf]);
							inspect_types.push_back(12);
						}

						// For each face of the T12 that matches a T1v or T2v face,
						// find the full tet (4 faces) from that T1/T2 and add them
						for (int lf = 0; lf < 4; ++lf) {
							auto fk = make_fk(tet.faces[lf][0], tet.faces[lf][1], tet.faces[lf][2]);
							int neighbor_type = 0;
							if (t1v_set.count(fk)) neighbor_type = 1;
							else if (t2v_set.count(fk)) neighbor_type = 2;

							if (neighbor_type > 0) {
								// Find the T1/T2 tet that contains this face
								const auto& src = (neighbor_type == 1) ? global_faces_t1_valid : global_faces_t2_valid;
								for (size_t ti = 0; ti + 3 < src.size(); ti += 4) {
									bool found = false;
									for (int ff = 0; ff < 4; ++ff) {
										auto fk2 = make_fk(src[ti+ff][0], src[ti+ff][1], src[ti+ff][2]);
										if (fk2 == fk) { found = true; break; }
									}
									if (found) {
										for (int ff = 0; ff < 4; ++ff) {
											inspect_faces.push_back(src[ti+ff]);
											inspect_types.push_back(neighbor_type);
										}
										break;
									}
								}
							}
						}

						if (!inspect_faces.empty()) {
							int nf = static_cast<int>(inspect_faces.size());
							Eigen::MatrixXi F(nf, 3);
							for (int i = 0; i < nf; ++i)
								F.row(i) << static_cast<int>(inspect_faces[i][0]),
									static_cast<int>(inspect_faces[i][1]),
									static_cast<int>(inspect_faces[i][2]);
							viewer_t12_inspect.data().set_mesh(global_V3d, F);
							viewer_t12_inspect.data().show_lines = true;
							viewer_t12_inspect.data().show_faces = true;
							Eigen::MatrixXd C(nf, 3);
						for (int i = 0; i < nf; ++i) {
							if (inspect_types[i] == 1)       C.row(i) << 0.4, 0.6, 1.0;
							else if (inspect_types[i] == 2)  C.row(i) << 0.4, 1.0, 0.4;
							else if (global_t12_inspect_is_removed) C.row(i) << 0.8, 0.2, 0.2;
							else                             C.row(i) << 1.0, 0.7, 0.3;
						}
							viewer_t12_inspect.data().set_colors(C);
						}
					}
				}
			}
		};

		// =========================
		// INTERACTIVE VALID CALLBACK
		// =========================
		menu_interactive.callback_draw_viewer_window = [&]() {
			static bool show_t1v = true;
			static bool show_t2v = true;
			static bool show_t12v = true;
			static bool show_t1r = false;
			static bool show_t2r = false;
			static bool show_t12r = false;

			ImGui::PushStyleColor(ImGuiCol_WindowBg, ImVec4(1.0f, 1.0f, 1.0f, 1.0f));
			ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.0f, 0.0f, 0.0f, 1.0f));
			ImGui::PushStyleColor(ImGuiCol_TitleBg, ImVec4(0.85f, 0.85f, 0.85f, 1.0f));
			ImGui::PushStyleColor(ImGuiCol_TitleBgActive, ImVec4(0.75f, 0.75f, 0.75f, 1.0f));

			ImGui::Begin("Interactive Tetrahedra");
			ImGui::SetWindowFontScale(1.4f);

			bool changed = false;
			bool filters_changed = false;

			ImGui::Text("Valid:");
			changed |= ImGui::Checkbox("T1 valid (blue)", &show_t1v);
			ImGui::SameLine();
			ImGui::TextColored(ImVec4(0.4f, 0.6f, 1.0f, 1.0f), "%d tets", delaunay3d_T1_valid);

			changed |= ImGui::Checkbox("T2 valid (green)", &show_t2v);
			ImGui::SameLine();
			ImGui::TextColored(ImVec4(0.4f, 0.6f, 1.0f, 1.0f), "%d tets", delaunay3d_T2_valid);

			changed |= ImGui::Checkbox("T12 valid (orange)", &show_t12v);
			ImGui::SameLine();
			ImGui::TextColored(ImVec4(0.4f, 0.6f, 1.0f, 1.0f), "%d tets", delaunay3d_T12_valid);

			ImGui::Separator();
			ImGui::Text("Removed:");
			changed |= ImGui::Checkbox("T1 removed (red)", &show_t1r);
			ImGui::SameLine();
			ImGui::TextColored(ImVec4(0.8f, 0.2f, 0.2f, 1.0f), "%d tets", delaunay3d_T1_removed);

			changed |= ImGui::Checkbox("T2 removed (red)", &show_t2r);
			ImGui::SameLine();
			ImGui::TextColored(ImVec4(0.8f, 0.2f, 0.2f, 1.0f), "%d tets", delaunay3d_T2_removed);

			changed |= ImGui::Checkbox("T12 removed (red)", &show_t12r);
			ImGui::SameLine();
			ImGui::TextColored(ImVec4(0.8f, 0.2f, 0.2f, 1.0f), "%d tets", delaunay3d_T12_removed);

			ImGui::Separator();
			ImGui::Text("T1 Filters:");
			filters_changed |= ImGui::Checkbox("T1 Midpoint filter", &global_filter_t1_midpoint);
			ImGui::SameLine();
			ImGui::TextColored(ImVec4(0.5f, 0.5f, 0.5f, 1.0f), "(removed %d)", global_t1_removed_by_midpoint);
			ImGui::Text("  Centroid removed: %d", global_t1_removed_by_centroid);

			ImGui::Separator();
			ImGui::Text("T2 Filters:");
			filters_changed |= ImGui::Checkbox("T2 Midpoint filter", &global_filter_t2_midpoint);
			ImGui::SameLine();
			ImGui::TextColored(ImVec4(0.5f, 0.5f, 0.5f, 1.0f), "(removed %d)", global_t2_removed_by_midpoint);
			ImGui::Text("  Centroid removed: %d", global_t2_removed_by_centroid);

			ImGui::Separator();
			ImGui::Text("T12 Filters:");
			filters_changed |= ImGui::Checkbox("T12 Midpoint filter", &global_filter_t12_midpoint);
			ImGui::SameLine();
			ImGui::TextColored(ImVec4(0.5f, 0.5f, 0.5f, 1.0f), "(removed %d)", global_t12_removed_by_midpoint);

			filters_changed |= ImGui::Checkbox("Isolated filter", &global_filter_t12_isolated);
			ImGui::SameLine();
			ImGui::TextColored(ImVec4(0.5f, 0.5f, 0.5f, 1.0f), "(removed %d)", global_t12_removed_by_isolated);

			filters_changed |= ImGui::Checkbox("IMA rescue", &global_filter_t12_ima_rescue);
			ImGui::SameLine();
			ImGui::TextColored(ImVec4(0.0f, 0.5f, 0.0f, 1.0f), "(rescued %d)", global_t12_rescued_by_ima);

			ImGui::Separator();

			int visible_faces = 0;
			if (show_t1v) visible_faces += static_cast<int>(global_faces_t1_valid.size());
			if (show_t2v) visible_faces += static_cast<int>(global_faces_t2_valid.size());
			if (show_t12v) visible_faces += static_cast<int>(global_faces_t12_valid.size());
			if (show_t1r) visible_faces += static_cast<int>(global_faces_t1_removed.size());
			if (show_t2r) visible_faces += static_cast<int>(global_faces_t2_removed.size());
			if (show_t12r) visible_faces += static_cast<int>(global_faces_t12_removed.size());
			ImGui::Text("Visible faces: %d", visible_faces);
			ImGui::Text("Total valid faces: %d", delaunay3d_all_valid_faces);
			ImGui::Text("Total T1: %d  T2: %d  T12: %d", (int)global_t1_tets.size(), (int)global_t2_tets.size(), (int)global_t12_tets.size());

			ImGui::Separator();
			ImGui::Text("Inspector:");
			if (global_t1t2_inspect_idx >= 0) {
				const char* type_name = (global_t1t2_inspect_type == 1) ? "T1" : "T2";
				if (global_t1t2_inspect_is_removed)
					ImGui::TextColored(ImVec4(0.8f, 0.2f, 0.2f, 1.0f), "Selected %s removed #%d", type_name, global_t1t2_inspect_idx);
				else
					ImGui::TextColored(ImVec4(0.4f, 0.6f, 1.0f, 1.0f), "Selected %s valid #%d", type_name, global_t1t2_inspect_idx);

				const auto& tets_list = (global_t1t2_inspect_type == 1) ? global_t1_tets : global_t2_tets;
				if (global_t1t2_inspect_idx < (int)tets_list.size()) {
					const auto& tet = tets_list[global_t1t2_inspect_idx];
					ImGui::Text("Centroid: (%.2f, %.2f) -> %s", tet.centroid_x, tet.centroid_y,
						tet.centroid_in_solid ? "SOLID" : "VOID");
					ImGui::Text("Base verts: [%d, %d, %d]", (int)tet.base_verts[0], (int)tet.base_verts[1], (int)tet.base_verts[2]);
					ImGui::Text("Apex vert: %d", (int)tet.apex_vert);
					ImGui::Separator();
					ImGui::Text("Edge samples (base->apex, t=2/5,1/2,3/5):");
					const char* sample_names2[3] = {"t=2/5", "t=1/2", "t=3/5"};
					for (int be = 0; be < 3; ++be) {
						ImGui::Text("  Edge [%d]->apex:", (int)tet.base_verts[be]);
						for (int si = 0; si < 3; ++si) {
							const char* l1_status = tet.midpoint_in_solid_l1[be][si] ? "SOLID" : "VOID";
							const char* l2_status = tet.midpoint_in_solid_l2[be][si] ? "SOLID" : "VOID";
							ImVec4 color = (tet.midpoint_in_solid_l1[be][si] || tet.midpoint_in_solid_l2[be][si])
								? ImVec4(0.0f, 0.5f, 0.0f, 1.0f) : ImVec4(0.8f, 0.0f, 0.0f, 1.0f);
							ImGui::TextColored(color, "    %s: (%.2f, %.2f) L1=%s L2=%s",
								sample_names2[si], tet.midpoint_x[be][si], tet.midpoint_y[be][si], l1_status, l2_status);
						}
					}
					ImGui::Text("Void count: L1=%d/9, L2=%d/9", tet.void_l1_count, tet.void_l2_count);
					if (tet.fails_midpoint)
						ImGui::TextColored(ImVec4(0.8f, 0.0f, 0.0f, 1.0f), "Midpoint filter: FAIL (all 9 void on both levels)");
					else
						ImGui::TextColored(ImVec4(0.0f, 0.5f, 0.0f, 1.0f), "Midpoint filter: PASS");
				}

				if (ImGui::Button("Clear selection")) {
					global_t1t2_inspect_idx = -1;
					global_t1t2_inspect_type = 0;
					global_t1t2_inspect_is_removed = false;
					global_t12_inspect_idx = -1;
					global_t12_inspect_is_removed = false;
					global_t12_inspect_needs_rebuild = true;
				}
			} else if (global_t12_inspect_idx >= 0) {
				if (global_t12_inspect_is_removed)
					ImGui::TextColored(ImVec4(0.8f, 0.2f, 0.2f, 1.0f), "Selected T12 removed #%d", global_t12_inspect_idx);
				else
					ImGui::TextColored(ImVec4(1.0f, 0.5f, 0.0f, 1.0f), "Selected T12 valid #%d", global_t12_inspect_idx);
				if (ImGui::Button("Clear selection")) {
					global_t12_inspect_idx = -1;
					global_t12_inspect_is_removed = false;
					global_t12_inspect_needs_rebuild = true;
					global_t1t2_inspect_idx = -1;
					global_t1t2_inspect_type = 0;
				}
			} else {
				ImGui::TextColored(ImVec4(0.5f, 0.5f, 0.5f, 1.0f), "Click a face to inspect");
			}

			ImGui::SetWindowFontScale(1.0f);
			ImGui::End();
			ImGui::PopStyleColor(4);

			if (filters_changed) {
				recompute_t1_t2_from_filters();
				recompute_t12_from_filters();
				changed = true;
			}

			if (changed || interactive_needs_rebuild) {
				interactive_needs_rebuild = false;
				viewer_interactive.data().clear();

				std::vector<std::array<index_t, 3>> all_f;
				std::vector<int> face_types;
				global_interactive_faces.clear();
				global_interactive_face_types.clear();
				if (show_t1v) {
					for (const auto& f : global_faces_t1_valid) { all_f.push_back(f); face_types.push_back(1); }
				}
				if (show_t2v) {
					for (const auto& f : global_faces_t2_valid) { all_f.push_back(f); face_types.push_back(2); }
				}
				if (show_t12v) {
					for (const auto& f : global_faces_t12_valid) { all_f.push_back(f); face_types.push_back(12); }
				}
				if (show_t1r) {
					for (const auto& f : global_faces_t1_removed) { all_f.push_back(f); face_types.push_back(-1); }
				}
				if (show_t2r) {
					for (const auto& f : global_faces_t2_removed) { all_f.push_back(f); face_types.push_back(-2); }
				}
				if (show_t12r) {
					for (const auto& f : global_faces_t12_removed) { all_f.push_back(f); face_types.push_back(-12); }
				}

				if (!all_f.empty() && global_V3d.rows() > 0) {
					int nf = static_cast<int>(all_f.size());
					Eigen::MatrixXi F(nf, 3);
					for (int i = 0; i < nf; ++i)
						F.row(i) << static_cast<int>(all_f[i][0]),
							static_cast<int>(all_f[i][1]),
							static_cast<int>(all_f[i][2]);
					viewer_interactive.data().set_mesh(global_V3d, F);
					viewer_interactive.data().show_lines = true;
					viewer_interactive.data().show_faces = true;
					Eigen::MatrixXd C(nf, 3);
					for (int i = 0; i < nf; ++i) {
						if (face_types[i] == 1)        C.row(i) << 0.4, 0.6, 1.0;
						else if (face_types[i] == 2)   C.row(i) << 0.4, 1.0, 0.4;
						else if (face_types[i] == 12)  C.row(i) << 1.0, 0.7, 0.3;
						else                           C.row(i) << 0.8, 0.2, 0.2;
					}
					viewer_interactive.data().set_colors(C);
				}
				global_interactive_faces = all_f;
				global_interactive_face_types = face_types;
			}
		};

		// Mouse click callback: pick T12 face in the interactive viewer
		viewer_interactive.callback_mouse_down = [&](igl::opengl::glfw::Viewer& vw, int button, int /*modifier*/) -> bool {
			if (button != 0) return false; // left click only
			if (global_interactive_faces.empty() || global_V3d.rows() == 0) return false;

			// Build F matrix from current interactive faces
			int nf = static_cast<int>(global_interactive_faces.size());
			Eigen::MatrixXi F(nf, 3);
			for (int i = 0; i < nf; ++i)
				F.row(i) << static_cast<int>(global_interactive_faces[i][0]),
					static_cast<int>(global_interactive_faces[i][1]),
					static_cast<int>(global_interactive_faces[i][2]);

			int fid = -1;
			Eigen::Vector3f bc;
			double x = vw.current_mouse_x;
			double y = vw.core().viewport(3) - vw.current_mouse_y;

			bool hit = igl::unproject_onto_mesh(
				Eigen::Vector2f(static_cast<float>(x), static_cast<float>(y)),
				vw.core().view, vw.core().proj, vw.core().viewport,
				global_V3d, F, fid, bc);

		if (hit && fid >= 0 && fid < (int)global_interactive_face_types.size()) {
				int face_type = global_interactive_face_types[fid];

				auto make_fk = [](index_t a, index_t b, index_t c) -> std::array<index_t, 3> {
					std::array<index_t, 3> f = {a, b, c};
					if (f[0] > f[1]) std::swap(f[0], f[1]);
					if (f[1] > f[2]) std::swap(f[1], f[2]);
					if (f[0] > f[1]) std::swap(f[0], f[1]);
					return f;
				};

				auto clicked_fk = make_fk(
					global_interactive_faces[fid][0],
					global_interactive_faces[fid][1],
					global_interactive_faces[fid][2]);

				// Handle T1/T2 clicks
				if (face_type == 1 || face_type == -1 || face_type == 2 || face_type == -2) {
					int tet_type = (face_type == 1 || face_type == -1) ? 1 : 2;
					bool is_removed = (face_type < 0);
					const auto& tets_list = (tet_type == 1) ? global_t1_tets : global_t2_tets;

					for (size_t i = 0; i < tets_list.size(); ++i) {
						for (int ff = 0; ff < 4; ++ff) {
							auto fk2 = make_fk(
								tets_list[i].faces[ff][0],
								tets_list[i].faces[ff][1],
								tets_list[i].faces[ff][2]);
							if (fk2 == clicked_fk) {
								global_t1t2_inspect_idx = static_cast<int>(i);
								global_t1t2_inspect_type = tet_type;
								global_t1t2_inspect_is_removed = is_removed;
								global_t12_inspect_idx = -1;
								global_t12_inspect_needs_rebuild = true;
								return true;
							}
						}
					}
				}

				// Handle T12 clicks
				if (face_type == 12 || face_type == -12) {
					bool is_removed = (face_type == -12);
					const auto& search_list = is_removed
						? global_faces_t12_removed : global_faces_t12_valid;

					for (size_t i = 0; i + 3 < search_list.size(); i += 4) {
						for (int ff = 0; ff < 4; ++ff) {
							auto fk2 = make_fk(
								search_list[i+ff][0],
								search_list[i+ff][1],
								search_list[i+ff][2]);
							if (fk2 == clicked_fk) {
								global_t12_inspect_idx = static_cast<int>(i / 4);
								global_t12_inspect_is_removed = is_removed;
								global_t12_inspect_needs_rebuild = true;
								global_t1t2_inspect_idx = -1;
								global_t1t2_inspect_type = 0;
								return true;
							}
						}
					}
				}
			}
			return false;
		};
	}











	// Load exported OBJ for the OBJ result viewer
	{
		std::string obj_path = "C:/Users/DELL/Desktop/Data/free_boundary.obj";
		Eigen::MatrixXd V_obj;
		Eigen::MatrixXi F_obj;
		if (igl::readOBJ(obj_path, V_obj, F_obj) && F_obj.rows() > 0) {
			viewer_obj_result.data().set_mesh(V_obj, F_obj);
			viewer_obj_result.data().show_lines = false;
			viewer_obj_result.data().show_faces = true;
			std::cout << "OBJ loaded for viewer: " << obj_path << " ("
				<< V_obj.rows() << " vertices, " << F_obj.rows() << " faces)\n";
		} else {
			viewer_obj_result.data().set_mesh(Vbox, Fbox);
			viewer_obj_result.data().show_faces = false;
			std::cerr << "Could not read OBJ: " << obj_path << "\n";
		}
	}

	// Collect enabled viewers
	std::vector<std::pair<igl::opengl::glfw::Viewer*, bool>> viewer_list;

	if (choice == 'Y' || choice == 'y') {
		// Multi-section mode: only contours + free boundary + OBJ result
		viewer_list = {
			{ &viewer_l1,           true },  // All contours
			{ &viewer_intersection, true },  // Free boundary
			{ &viewer_obj_result,   ENABLE_VIEWER_OBJ_RESULT },
		};
	} else {
		// Single-pair mode: use ENABLE flags
		viewer_list = {
			{ &viewer_l1,            ENABLE_VIEWER_L1 },
			{ &viewer_l2,            ENABLE_VIEWER_L2 },
			{ &viewer_intersection,  ENABLE_VIEWER_INTERSECTION },
			{ &viewer_overlay,       ENABLE_VIEWER_OVERLAY },
			{ &viewer_hh,            ENABLE_VIEWER_HH },
			{ &viewer_hs,            ENABLE_VIEWER_HS },
			{ &viewer_vor_l1,        ENABLE_VIEWER_VOR_L1 },
			{ &viewer_vor_l2,        ENABLE_VIEWER_VOR_L2 },
			{ &viewer_ima_l1,        ENABLE_VIEWER_IMA_L1 },
			{ &viewer_ema_l1,        ENABLE_VIEWER_EMA_L1 },
			{ &viewer_ima_l2,        ENABLE_VIEWER_IMA_L2 },
			{ &viewer_ema_l2,        ENABLE_VIEWER_EMA_L2 },
			{ &viewer_both_l1,       ENABLE_VIEWER_BOTH_L1 },
			{ &viewer_both_l2,       ENABLE_VIEWER_BOTH_L2 },
			{ &viewer_proj_l1_on_l2, ENABLE_VIEWER_PROJ_L1_ON_L2 },
			{ &viewer_proj_l2_on_l1, ENABLE_VIEWER_PROJ_L2_ON_L1 },
			{ &viewer_delaunay3d,    ENABLE_VIEWER_DELAUNAY3D },
			{ &viewer_t1,            ENABLE_VIEWER_T1 },
			{ &viewer_t2,            ENABLE_VIEWER_T2 },
			{ &viewer_t12,           ENABLE_VIEWER_T12 },
			{ &viewer_t1_valid,      ENABLE_VIEWER_T1_VALID },
			{ &viewer_t1_removed,    ENABLE_VIEWER_T1_REMOVED },
			/*{&viewer_t2_valid,      ENABLE_VIEWER_T2_VALID},*/
			{ &viewer_t2_removed,    ENABLE_VIEWER_T2_REMOVED },
			{ &viewer_t12_valid,     ENABLE_VIEWER_T12_VALID },
			{ &viewer_t12_removed,   ENABLE_VIEWER_T12_REMOVED },
			{ &viewer_all_valid,     ENABLE_VIEWER_ALL_VALID },
			{ &viewer_all_valid_gray, ENABLE_VIEWER_ALL_VALID_GRAY },
			{ &viewer_free_boundary,  ENABLE_VIEWER_FREE_BOUNDARY },
			{ &viewer_interactive,   ENABLE_VIEWER_INTERACTIVE },
			{ &viewer_t12_inspect,   ENABLE_VIEWER_T12_INSPECT },
			{ &viewer_obj_result,    ENABLE_VIEWER_OBJ_RESULT },
		};
	}

	// Find the last enabled viewer (it will get the blocking launch)
	int last_enabled = -1;
	for (int i = static_cast<int>(viewer_list.size()) - 1; i >= 0; --i) {
		if (viewer_list[i].second) { last_enabled = i; break; }
	}

	if (last_enabled < 0) {
		std::cerr << "No viewers enabled. Exiting.\n";
		return 0;
	}

	for (int i = 0; i < static_cast<int>(viewer_list.size()); ++i) {
		if (!viewer_list[i].second) continue;
		if (i == last_enabled)
			viewer_list[i].first->launch();      // blocking
		else
			viewer_list[i].first->launch(false); // non-blocking
	}

	return 0;
}
	