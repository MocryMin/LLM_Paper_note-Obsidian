# Introduction to Notebook

This is a personal notebook written when reading papers on LLM, ranging from model architecture, fine-tuning, agent, *RAG, memory system(which mainly belong to agent but also involve other sections)*.  

## Structure  

**Notebook Structure**

The structure of notebook is designed as a **graph**, where each node is a *paper* or a *conclusive knowledge patch(e.g. word2vec)*. Every node will be linked to other nodes in the following two ways:  
1. **study timeline pointer**: The timeline I read all the papers. The previous paper usually contain some prerequisites to the later ones. Since I asked AI to plan the reading route according to the dependency between different papers, the sub-graph connected by timeline pointer(timeline graph) generally formed a DAG, following which you are assumably to encounter less obstacles while reading these complicated papers. **timeline pointer are explicitly given in the header of each note.**  	
2. **relevance pointer**: This pointer exits when I associate something in a new paper to that in previous ones, or when I come up which a new problem that might be solved in a future-to-read paper. **relevance pointer always exits inside the note.**  

**Note Structure**

The structures note pieces(usually a *.md*) are always not the same, following a casual mindflow. But their containing sections can be generally concluded as following: 
1. **note header**: Exists at the beginning of each note. Marking the citation of original paper(reference to its web space), reading time and the timeline pointer.
2. **mindflow**: In not fixed format. Just some *taking-downs* of minds that spring up when going through the paper.
3. ***items***: Usually something important in the paper. You will recognize them when you encounter them.
4. **worth noticing**: Something less important in the paper, but somehow inspires or catches my attention.

## Tags

I have used "#" tag format which is supported by obsidian in my notes. There are currently two types of tags in my notebook:
1. **Direction Tag**: This kind of tag only exists right at the beginning of every note. They mark the direction of the paper in this note, "architecture", "agent"...for example. To give a quick location of a paper or serve the purpose of tag query.
2. **LazyTag**: This tag can exist anywhere in a note, marking an interesting or potentially meaningful knowledge which I currently have no extra time to learn. Via lazytag, I can easily call them back when I have vacancy. 