#include "view.h"
#include <igl/project.h>

Eigen::Vector2f projectToScreen(
	const Eigen::Vector3f& p,
	const igl::opengl::glfw::Viewer& viewer)
{
	Eigen::Vector4f viewport(
		0, 0,
		viewer.core().viewport[2],
		viewer.core().viewport[3]
	);

	Eigen::Vector3f screen =
		igl::project(
			p,
			viewer.core().view,
			viewer.core().proj,
			viewport
		);

	return Eigen::Vector2f(screen.x(), viewport[3] - screen.y());
}
