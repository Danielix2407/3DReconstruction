

#ifndef __POINTSET_H__
#define __POINTSET_H__

#include <vector>
#include <Eigen/Core>

//---------------------------------------------------------------------------------------//
//---------- 							D E F I N E S 							---------//
//---------------------------------------------------------------------------------------//
#define	INSIDE	21

//---------------------------------------------------------------------------------------//
//---------					  D A T A   T Y P E   D E C L S				        ---------//
//---------------------------------------------------------------------------------------//

//---------------------------------------------------------------------------------------//
//	STRUCTURE		:	[DataPoints]
//	MADE BY			:	[JSC] Jefferson Salcedo Chavez		DATE:	FEB/2026
//	DESCRIPTION		:	Structure to keep together the points.  
//---------------------------------------------------------------------------------------//
struct datapoints
{
	std::vector<Eigen::Vector3d> points;
	double bbox_min_x=0,bbox_min_y=0,bbox_max_x=0,bbox_max_y=0;
	bool bbox_valid=false;
	void compute_bbox(){
		if(points.empty()){bbox_valid=false;return;}
		bbox_min_x=bbox_max_x=points[0].x();
		bbox_min_y=bbox_max_y=points[0].y();
		for(size_t i=1;i<points.size();++i){
			double x=points[i].x(),y=points[i].y();
			if(x<bbox_min_x)bbox_min_x=x;else if(x>bbox_max_x)bbox_max_x=x;
			if(y<bbox_min_y)bbox_min_y=y;else if(y>bbox_max_y)bbox_max_y=y;
		}
		bbox_valid=true;
	}
};

//---------------------------------------------------------------------------------------//
//	FUNCION NAME	:	[points_resample]
//  MADE BY:			[JSC] Jefferson Salcedo Chavez				DATE: FEB/2003
//	DESCRIPTION		:	Resamples the points, aqui se esta definiendo una distancia maxima
//						y a partir de aqui se puede dividir en 2 o 3, o nose como mas hacerlo
//---------------------------------------------------------------------------------------//
/*void
points_resample
(
	polyline* poly,			// <> Polyline to be resampled aqui debe ir el segmento 
	double		dist,			// <  max. length for each edge aqui va la distancia de ese segmento
	polyline* rpoly			//  > resampled polyline aqui salen los puntos mas el resample
);
*/
//---------------------------------------------------------------------------------------//
//	FUNCION NAME	:	[polyline_invertPoints]
//  MADE BY:			[JSC] Jefferson Salcedo Chavez			DATE: FEB/2026
//	DESCRIPTION		:	Aqui podria invertir orden a los que sean huecos
//---------------------------------------------------------------------------------------//
/*void
polyline_invertPoints
(
	polyline* poly			// <  Polyline
);
*/
//---------------------------------------------------------------------------------------//
//	FUNCION NAME	:	[polyline_calcMinMax]
//  MADE BY:			[EVO] Jefferson Salcedo Chavez				DATE: FEB/2026
//	DESCRIPTION		:	Calcula el min y maximo supongo para despues dar la distancia maxima y hacer el resample
//---------------------------------------------------------------------------------------//
/*void
polyline_calcMinMax
(
	polyline* poly,			// <  Polyline aqui van los puntos originales sin resample
	vector3d	min,			//  > min point
	vector3d	max				//  > max point
);
*/
//---------------------------------------------------------------------------------------//
//	FUNCION NAME	:	[polyline_findClosestPoint]
//  MADE BY:			[EVO] Eliana Vásquez Osorio				DATE: FEB/2003
//	DESCRIPTION		:	Find the index of the closests point (in poly->points) to point
//---------------------------------------------------------------------------------------//
/*void
polyline_findClosestPoint
(
	polyline* poly,			// <  Polyline
	vector3d	point,
	int* index
);
*/
#endif