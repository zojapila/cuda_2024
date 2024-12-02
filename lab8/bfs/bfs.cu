#include "bfsFiles.h"
#include <stdio.h>
#include <stdlib.h>

#define BLOCK_SIZE 128
// maximum number of elements that can be inserted into a block queue
#define BLOCK_QUEUE_SIZE 8192

void BFSSequential(int source, int *edges, int *dest, int *label, int nodes)
{
	int *cFrontier = (int *)malloc(nodes * sizeof(int));
	int cFrontierTail = 0;
	int *pFrontier = (int *)malloc(nodes * sizeof(int));
	int pFrontierTail = 0;

	pFrontier[pFrontierTail++] = source;
	while (pFrontierTail > 0)
	{
		// visit all previous frontier vertices
		for (int f = 0; f < pFrontierTail; f++)
		{
			// pick up one of the previous frontier vertices
			int cVertex = pFrontier[f];
			// for all its edges
			for (int i = edges[cVertex]; i < edges[cVertex + 1]; i++)
			{
				// the dest vertex has not been visited
				if (label[dest[i]] == -1)
				{
					cFrontier[cFrontierTail++] = dest[i];
					label[dest[i]] = label[cVertex] + 1;
				}
			}
		}
		// swap previous and current
		int *temp = cFrontier;
		cFrontier = pFrontier;
		pFrontier = temp;
		pFrontierTail = cFrontierTail;
		cFrontierTail = 0;
	}

	free(cFrontier);
	free(pFrontier);
}

__global__ void BFSKernelGlobalQueue(int *edges, int *dest, int *label, int *pFrontier, int *cFrontier, int *pFrontierTail, int *cFrontierTail)
{
	//@@ INSERT KERNEL CODE HERE
}

__global__ void BFSKernelBlockQueue(int *edges, int *dest, int *label, int *pFrontier, int *cFrontier, int *pFrontierTail, int *cFrontierTail)
{
	//@@ INSERT KERNEL CODE HERE
}

void BFSHost(int mode, int *h_edges, int *h_dest, int *h_label, int nodes, int *h_pFrontier, int *h_pFrontierTail)
{
	//@@ INSERT HOST CODE HERE

	// allocate edges, dest, label in device global memory
	// allocate pFrontier, cFrontier, cFrontierTail, pFrontierTail in device global memory

	// copy the data from host to device

	// launch a kernel in a loop
	while ()
	{
		if (mode == 0)
			BFSKernelGlobalQueue<<<>>>();
		else if (mode == 1)
			BFSKernelBlockQueue<<<>>>();

		// read the current frontier and copy it from device to host

		// swap the roles of the frontiers

		// set pFrontierTail and cFrontierTail
	}

	// copy data to label

	// free device memory
}

int main(int argc, char *argv[])
{
	// check if number of input args is correct: input and output image filename
	if (argc != 5)
	{
		printf("Wrong number of arguments: exactly 4 arguments needed (1 input and 3 output .txt filenames, suggested: out_seq.txt, out_global.txt, out_block.txt)\n");
		return 1;
	}

	int source;
	int *edges;
	int *dest;
	int *labelSeq;
	int *labelGlobal;
	int *labelBlock;
	int nodes;
	int *pFrontierGlobal;
	int *pFrontierBlock;
	int pFrontierTailGlobal;
	int pFrontierTailBlock;

	int status = readFile(argv[1], &source, &edges, &dest, &labelSeq, &labelGlobal, &labelBlock, &nodes, &pFrontierGlobal, &pFrontierBlock, &pFrontierTailGlobal, &pFrontierTailBlock);
	if (status != SUCCESS)
	{
		printf("Cannot read from file!\n");
		return 2;
	}

	BFSSequential(source, edges, dest, labelSeq, nodes);
	BFSHost(0, edges, dest, labelGlobal, nodes, pFrontierGlobal, &pFrontierTailGlobal);
	BFSHost(1, edges, dest, labelBlock, nodes, pFrontierBlock, &pFrontierTailBlock);

	writeFile(argv[2], &labelSeq, &nodes);
	writeFile(argv[3], &labelGlobal, &nodes);
	writeFile(argv[4], &labelBlock, &nodes);

	free(pFrontierBlock);
	free(pFrontierGlobal);
	free(labelBlock);
	free(labelGlobal);
	free(labelSeq);
	free(dest);
	free(edges);

	return 0;
}
