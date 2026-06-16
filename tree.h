

#ifndef __TREE_H__
#define __TREE_H__

#include <vector>
#include <string>

//---------------------------------------------------------------------------------------//
//---------- 							D E F I N E S 							---------//
//---------------------------------------------------------------------------------------//

//---------------------------------------------------------------------------------------//
//---------					  D A T A   T Y P E   D E C L S				        ---------//
//---------------------------------------------------------------------------------------//

//---------------------------------------------------------------------------------------//
//	STRUCTURE		:	[tree_node]
//	MADE BY			:	[JSC] Jefferson Salcedo Chavez		DATE:	FEB/2026
//	DESCRIPTION		:	 
//---------------------------------------------------------------------------------------//
struct tree_node
{
	int contourId;
	int depth;
	int parent;
	std::vector<int> children;	
};

//---------------------------------------------------------------------------------------//
//	FUNCION NAME	:	[contourDepth]
//  MADE BY:			[JSC] Jefferson Salcedo Chavez				DATE: FEB/2026
//	DESCRIPTION		:	Calculates the depth of a node in the forest
//---------------------------------------------------------------------------------------//
int
	contourDepth
	(
		int								node,		// < Node index
		const std::vector<tree_node>&	forest		// < Forest structure
	);

//---------------------------------------------------------------------------------------//
//	FUNCION NAME	:	[printTree]
//  MADE BY:			[JSC] Jefferson Salcedo Chavez				DATE: FEB/2026
//	DESCRIPTION		:	Prints the tree structure to console recursively
//---------------------------------------------------------------------------------------//
void
	printTree
	(
		const std::vector<tree_node>&	forest,		// < Forest structure
		int								node,		// < Current node index
		int								depth = 0	// < Current depth for indentation
	);

//---------------------------------------------------------------------------------------//
//	FUNCION NAME	:	[format_tree_structure]
//  MADE BY:			[JSC] Jefferson Salcedo Chavez				DATE: FEB/2026
//	DESCRIPTION		:	Formats a tree structure recursively into a string
//---------------------------------------------------------------------------------------//
std::string
	format_tree_structure
	(
		const std::vector<tree_node>&	forest,			// < Forest structure
		int								node,			// < Current node index
		bool							useLetters		// < Use letters (true) or numbers (false)
	);

//---------------------------------------------------------------------------------------//
//	FUNCION NAME	:	[format_forest_text]
//  MADE BY:			[JSC] Jefferson Salcedo Chavez				DATE: FEB/2026
//	DESCRIPTION		:	Formats the entire forest into a text string
//---------------------------------------------------------------------------------------//
std::string
	format_forest_text
	(
		const std::vector<tree_node>&	forest,			// < Forest structure
		bool							useLetters		// < Use letters (true) or numbers (false)
	);

#endif