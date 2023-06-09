import numpy as np
import heapq
import functools
from math import floor, ceil


class IntervalSet:
    def __init__(self, edges):
        self.edges = edges

    def to_set(self):
        return np.hstack([np.arange(a, b + 1) for a, b in self.edges])

    def contains(self, edge):
        """Determines if self - edge == self."""
        lo, hi = edge

        if hi - lo <= 1:
            return True

        def no_change(start, end):
            if end <= lo or hi <= start:
                return True

        return all(no_change(*edge) for edge in self.edges)

    def __sub__(self, edge):
        edges = [x for x in self.edges]
        lo, hi = edge
        for i, (start, end) in enumerate(self.edges):
            if end < lo or hi < start:
                # no intersection
                continue

            lo_in_range = start <= lo <= end
            hi_in_range = start <= hi <= end

            if lo_in_range and hi_in_range:
                # interval to remove entirely contained
                edges = edges[:i] + [(start, lo), (hi, end)] + edges[i + 1 :]
                return IntervalSet(edges)
            elif lo_in_range and not hi_in_range:
                # start - lo - end - hi
                # replace [start, end] with [start, lo] and then sub [end, hi]
                edges[i] = (start, lo)
            elif not lo_in_range and hi_in_range:
                # lo - start - hi - end
                # same as above
                edges[i] = (hi, end)
            else:
                # we already dealt with no overlap, so this must be containment
                # we can completely replace
                edges[i] = (lo, hi)
                return IntervalSet(edges)

        return IntervalSet(edges)


class RangeMinQuery:
    def __init__(self, arr):
        self.arr = [(x, i) for i, x in enumerate(arr)]
        self.block_size = ceil(np.log2(len(arr)) / 4)
        self.blocks = [min(self.arr[i:i+self.block_size]) for i in range(0, len(self.arr), self.block_size)]

    def min_range(self, i, j):
        mid_block = self.blocks[ceil(i / self.block_size) : floor(j / self.block_size)]
        start_block = self.arr[i:min(j + 1, ceil(i / self.block_size) * self.block_size)]
        end_block = self.arr[max(i, floor(j / self.block_size) * self.block_size):j+1]         
                
        return min(start_block + mid_block + end_block)[1]


class Arc:
    """A potential h-arc."""

    def __init__(self, v1, v2, parent, children):
        self.v1 = v1
        self.v2 = v2
        self.parent = parent
        self.children = children
        self.cost = 0
        self.cutoff = 0
        self.processed = False        
        self.l2 = []

    def __lt__(self, other):
        return -self.cutoff < -other.cutoff
        # return (-self.cutoff, self.v1, self.v2) < (-other.cutoff, other.v1, other.v2)


def frac(num, denom):
    """
    Representation of fractions. fractions.Fraction can be used to
    ensure rounding errors don't occur for all-integer inputs, whereas
    floating-point is accurate enough for most conceivable use-cases and
    can accept non-integral inputs (e.g., to rescale the input
    dimensions.)
    """
    # alternative: return Fraction(num, denom)
    return num / denom


def flatten_verts(arcs):
    """Takes in a list of arcs and returns a list of vertices
    (v1, v2, v1, v2), etc."""
    ans = []
    for arc in arcs:
        ans.append(arc.v1)
        ans.append(arc.v2)
    return ans


def postorder_traversal(f, leaves):
    """Computes f(root), where f depends on the children.
    Uses iteration to avoid stack overflow."""
    processed = set()
    frontier = set(leaves)
    while frontier:
        node = frontier.pop()
        if set(node.children).issubset(processed):
            result = f(node)
            processed.add(node)

            if node.parent is not None:
                frontier.add(node.parent)

    return result

def optimal_cost_root_harc(dims):
    """Assumes dims[0], dims[-1] are two smallest elements."""
    arcs = []
    leaves = []

    rmq = RangeMinQuery(dims)

    def make_arc(edge):
        """
        Initializes Arcs starting at the given edge, adding it to leaves
        if it's a leaf.
        """
        i1, i2 = edge        
        if i2 - i1 <= 1:
            # leaf node
            children = []
        elif i2 - i1 == 2:
            # degenerate leaf node
            leaf = sorted((max((i1, i2), key=lambda x: dims[x]), i1 + 1))
            children = [make_arc(leaf)]
        else:
            k = rmq.min_range(i1 + 1, i2 - 1)
            children = []
            for e in ((i1, k), (k, i2)):
                if e[1] - e[0] > 1:
                    children.append(make_arc(e))

        arc = Arc(i1, i2, None, children)
        arcs.append(arc)
        if not children:
            leaves.append(arc)

        for child in children:
            child.parent = arc

        return arc

    root = (0, len(dims) - 1)
    root = make_arc(root)

    edge_weights = np.zeros_like(dims)
    edge_weights[1:] = np.cumsum(dims[1:] * dims[:-1])

    @functools.lru_cache
    def W(e1, e2):
        if e2 <= e1:
            return 0

        return edge_weights[e2] - edge_weights[e1]

    def fan_cost(e1, e2, *edges, exclude=()):
        if e1 in exclude:
            hull = [e1 + 1, e2]
        elif e2 in exclude:
            hull = [e1, e2 - 1]
        else:
            hull = [e1, e2]

        iset = IntervalSet([hull])

        for i in range(0, len(edges), 2):
            iset -= edges[i : i + 2]

        total = 0

        prev = None
        hull_weight = 0
        for i, (a, b) in enumerate(iset.edges):
            if i >= 1 and a != prev:
                hull_weight += dims[a] * dims[prev]

            prev = b
            total += W(a, b)

        result = total + hull_weight
        return result

    def compute_ceiling(node, arcs, exclude=()):
        """Computes the ceiling, or covering set, of arcs above node."""
        iset = IntervalSet([(node.v1, node.v2)])
        ceiling = []
        for edge in sorted(arcs):
            if (
                not iset.contains((edge.v1, edge.v2)) 
                and edge not in ceiling
                and edge not in exclude
            ):
                iset -= (edge.v1, edge.v2)
                ceiling.append(edge)

        return ceiling    

    def compute_cutoff(node):
        if len(node.children) == 0:
            leaf = node
            leaf.cost = 0
            leaf.cutoff = 0
            am1, a = sorted((leaf.v1, leaf.v2), key=lambda i: dims[i])
            ap1 = min([leaf.parent.v1, leaf.parent.v2], key=lambda i: dims[i])

            leaf.parent.cost = dims[am1] * dims[ap1] * dims[a]
            leaf.parent.cutoff = frac(
                dims[am1] * dims[a] * dims[ap1],
                dims[a] * (dims[am1] + dims[ap1]) - dims[am1] * dims[ap1],
            )
            leaf.l2 = [leaf]
            leaf.processed = True            
        else:
            vmin = node.v1 if dims[node.v1] <= dims[node.v2] else node.v2        

            l2prime = []
            for child in node.children:
                l2prime += [x for x in child.l2]

            l2prime.sort()

            while l2prime and l2prime[0].cutoff >= dims[vmin]:
                heapq.heappop(l2prime)

            ceiling = compute_ceiling(node, l2prime)

            if node.cost == 0:
                # T(r1), T(r2_i), and fan of r0 under r1 and r2_i centered at vmin
                node.cost = sum([x.cost for x in ceiling]) + dims[vmin] * fan_cost(
                    node.v1, node.v2, *flatten_verts(ceiling), exclude=(vmin,),
                )

            def c1(w_v):
                return node.cost + w_v * dims[node.v1] * dims[node.v2]

            # print(node, [float(x.cutoff) for x in l2prime])
            node.l2 = [x for x in l2prime]        

            node.cutoff = dims[vmin]
            prev_fan_w = None
            fan_w = None
            ceiling_cost = None
            cutoff = dims[vmin]
            ceilings = []
            diffs = []
            cutoffs = []

            while len(ceiling) >= 0:
                # cost of a fan bounded by the ceiling

                prev_ceiling_cost = ceiling_cost
                ceiling_cost = sum([x.cost for x in ceiling])
                prev_fan_w = fan_w
                fan_w = fan_cost(node.v1, node.v2, *flatten_verts(ceiling))
                c2 = ceiling_cost + cutoff * fan_w

                ceilings.append([x for x in ceiling])
                diffs.append((c2, c1(cutoff)))
                cutoffs.append(cutoff)
                if c2 <= c1(cutoff):
                    break
                else:
                    if len(ceiling) == 0:
                        break
                    lowest_i = max(range(len(ceiling)), key=lambda i: ceiling[i].cutoff)
                    lowest = ceiling[lowest_i]
                    # print(ceiling, lowest)
                    prev_cutoff = cutoff
                    cutoff = lowest.cutoff
                    # remove lowest edge
                    ceiling = ceiling[:lowest_i] + ceiling[lowest_i + 1 :]
                    if len(lowest.children) == 0:
                        continue

                    new_edges = compute_ceiling(node, lowest.l2[1:], exclude=ceiling)
                    ceiling += new_edges
            if len(ceiling) == 0 and c2 > c1(cutoff):
                # optimal cutoff has no other h-arcs
                node.cutoff = frac(
                    node.cost, fan_cost(node.v1, node.v2) - dims[node.v1] * dims[node.v2]
                )
            elif prev_fan_w is None:
                # this cutoff is higher than all previous
                node.cutoff = frac(
                    node.cost - ceiling_cost, fan_w - dims[node.v1] * dims[node.v2]
                )
            else:
                # correct answer is between current ceiing and previous
                node.cutoff = frac(
                    node.cost - prev_ceiling_cost,
                    prev_fan_w - dims[node.v1] * dims[node.v2],
                )
                assert cutoff <= node.cutoff <= prev_cutoff

            assert node.cutoff <= dims[vmin]

            while node.l2 and node.l2[0].cutoff >= node.cutoff:
                heapq.heappop(node.l2)

            heapq.heappush(node.l2, node)            

        return node

    root = postorder_traversal(compute_cutoff, leaves)
    return (root, root.cost, arcs, leaves)


def optimal_matrix_chain_cost(dims):
    """Finds the optimal order and cost for multiplying matrices with the
    dimensions given by dims."""

    # It makes things easier if a single horizontal arc is the root of the
    # entire arc tree. We can transform any problem into one of this kind as
    # follows:
    # 1. Roll the array so the minimum value is at the front. This doesn't
    #    change the answer (once we unroll at the end), because the polygon
    #    remains unchanged.
    # 2. Add a new vertex equal to that minimum at the very end. This
    # introduces a single additional triangle which we can remove at the
    # end. Now (0, n - 1) is a horizontal arc that spans the entire polygon.

    offset = np.argmin(dims)
    dims = np.roll(dims, -offset, axis=0)
    dims = np.hstack([dims, dims[0]])

    root, root_cost, arcs, leaves = optimal_cost_root_harc(dims)
    cost = root.cost - dims[0] ** 2 * min(dims[1:-1])
    return (root, cost, arcs, leaves)


def multi_dot(mats, dot=np.dot):
    dims = [mat.shape[0] for mat in mats]
    dims.append(mats[-1].shape[1])


    offset = np.argmin(dims)
    dims2 = np.roll(dims, -offset, axis=0)
    dims2 = np.hstack([dims2, dims2[0]])

    root, cost, arcs, leaves = optimal_matrix_chain_cost(dims)
    def find_ceiling(node):
        if len(node.children) == 0:
            node.ceiling = set()
            node.matrix_product = None
        else:
            vmin = node.v1 if dims2[node.v1] <= dims2[node.v2] else node.v2
            wmin = dims2[vmin]
            frontier = set(node.children)
            node.ceiling = set()
            jumps = {}
            rjumps = {}

            while frontier:
                child = frontier.pop()
                if child.cutoff <= wmin:
                    node.ceiling.add(child)
                else:
                    frontier.update(child.ceiling)

        return node.ceiling
    
    ceiling = postorder_traversal(find_ceiling, leaves)
    def triangulate(node):
        if node.v2 - node.v1 == 1:
            node.triangulation = set()
            return node.triangulation
        elif node.v2 - node.v1 == 2:
            v3 = node.v1 + 1
            node.triangulation = {(node.v1, node.v2, v3)}
            return node.triangulation
        
        jumps = {}
        rjumps = {}
        for e in node.ceiling:
            jumps[e.v1] = (e, e.v2)
            rjumps[e.v2] = (e, e.v1)
        
        node.triangulation = []
        if dims2[node.v1] <= dims2[node.v2]:                
            vstart = node.v1
            vend = node.v2
            incr = 1
            edge_dict = jumps
        else:        
            vstart = node.v2  
            vend = node.v1
            incr = -1        
            edge_dict = rjumps

        curr_edges = set()
        curr = vstart        
        while curr != vend:            
            if curr in edge_dict:
                (edge, dest) = edge_dict[curr]         
                curr_edges.update(edge.triangulation)
                curr_edges.add(tuple(sorted((vstart, curr, dest))))
                curr = dest
            else:
                curr_edges.add(tuple(sorted((curr, curr + incr, vstart))))
                curr = curr + incr

        node.triangulation = curr_edges
        return node.triangulation
    import itertools
    from collections import defaultdict

    m = postorder_traversal(triangulate, leaves)
    triangles = []    
    for edge in m:    
        relabeled = [(v + offset) % len(dims) for v in edge]    
        v1, v2, v3 = relabeled
        print(relabeled)
        if len(set(relabeled)) == 3:        
            triangles.append(tuple(sorted(relabeled)))
    edges = []
    for (v1, v2, v3) in triangles:
        edges += [(v1, v2), (v1, v3), (v2, v3)]
    graph = defaultdict(set)
    for (v1, v2) in edges:
        graph[v1].add(v2)
        graph[v2].add(v1)

    tris = set()
    for v, neighbors in graph.items():
        for i, j in itertools.combinations(neighbors, 2):
            if j in graph[i]:
                tris.add(tuple(sorted((v, i, j))))

    products = {}
    for (v1, v2, v3) in triangles:
        for va, vb in ((v1, v2), (v2, v3)):
            if vb - va == 1:
                products[(va, vb)] = mats[va]


    while triangles:
        new_triangles = []
        for (v1, v2, v3) in triangles:
            if (v1, v2) in products and (v2, v3) in products:                
                products[(v1, v3)] = dot(products[(v1, v2)], products[(v2, v3)])
            else:
                new_triangles.append((v1, v2, v3))
        
        triangles = new_triangles

    return products[(0, len(dims) - 1)]