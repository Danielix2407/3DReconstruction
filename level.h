

#ifndef __LEVEL_H__
#define __LEVEL_H__

#include <vector>
#include "pointset.h"

//---------------------------------------------------------------------------------------//
//---------- 							D E F I N E S 							---------//
//---------------------------------------------------------------------------------------//
#define	INSIDE	21

//---------------------------------------------------------------------------------------//
//---------					  D A T A   T Y P E   D E C L S				        ---------//
//---------------------------------------------------------------------------------------//

//---------------------------------------------------------------------------------------//
//	STRUCTURE		:	[level_info]
//	MADE BY			:	[JSC] Jefferson Salcedo Chavez		DATE:	FEB/2026
//	DESCRIPTION		:	Structure to keep together the points.  
//---------------------------------------------------------------------------------------//
struct level_info
{
	std::vector<datapoints> contours;  // Conjunto de contornos del nivel
	std::vector<int> Ex_In_contour;    // Información exterior/interior
	double zCoord;                     // Coordenada Z del plano
	int num_contours;                  // Número de contornos en el nivel
};

//---------------------------------------------------------------------------------------//
//	FUNCION NAME	:	[readCnt]
//  MADE BY:			[JSC] Jefferson Salcedo Chavez				DATE: FEB/2026
//	DESCRIPTION		:	Reads a .cnt file and returns a vector of level_info
//---------------------------------------------------------------------------------------//
bool
	readCnt
	(
		const std::string&				filename,	// < Path to the .cnt file
		std::vector<level_info>&		levels		// < Output vector of levels
	);

#endif