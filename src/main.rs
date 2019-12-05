use std::collections::{ HashMap, HashSet, BinaryHeap };
use std::cmp::{ Ordering };

fn main(){
    let mut s = Vec::new();
//    s.push(Bytes::from(&b"abcba"[..]));
//    s.push(Bytes::from(&b"abbac"[..]));
    s.push("abcba".chars().collect::<Vec<char>>());
    s.push("abbac".chars().collect::<Vec<char>>());
    println!("{:?}", s);

    let tries = Trie::new(s);

    let fa = FactorOracle::new(tries);
    println!("{:?}", fa.states);
    println!("{:?}", fa.state_set_tree.print(0));
    println!("--------------------------------");
    println!("{:?}", fa.trans);

    //fa.state_set_tree.print(0);
    //let mut searchable: HashSet<usize> = HashSet::new();
    //searchable.insert(6);
    //println!("{:?}", fa.state_set_tree.search(&searchable));
    let hc = HuffmanCode::new(fa.occurence);
    hc.print_code_info();

    let mut cfa: Vec<i128> = Vec::new();
    cfa.push(0);
    
}

struct HuffmanCode {
    code_info: HashMap<char, usize>
}

impl HuffmanCode {
    pub fn new(data: HashMap<char, i64>) -> HuffmanCode {
        let mut queue: BinaryHeap<TreeNode> = BinaryHeap::new();
        for (key, val) in data.iter() {
            queue.push(TreeNode::new((*key, -*val, Vec::with_capacity(2))));
        }
        while queue.len() > 1 {
            let n: (TreeNode, TreeNode) = (queue.pop().unwrap(), queue.pop().unwrap());
            queue.push(TreeNode::new((
                            ' ',
                            n.0.occurence + n.1.occurence,
                            vec![n.0, n.1]
                        )));
        }
        let root: TreeNode = queue.pop().unwrap();
        let mut code_info: HashMap<char, usize> = HashMap::new();
        dfs(&mut code_info, &root, 0, 0);
        HuffmanCode {
            code_info: code_info
        }
    }

    fn print_code_info(&self) {
        for (k, v) in self.code_info.iter() {
            println!("{}: {:0b}", k, v);
        }
    }
}

fn dfs(code_info: &mut HashMap<char, usize>, node: &TreeNode, code: usize, code_size: usize) {
    if !node.childs.is_empty() {
        dfs(code_info, &node.childs[0], code << 1, code_size+1);
        dfs(code_info, &node.childs[1], (code << 1) + 1, code_size+1);
    } else {
        code_info.insert(node.value, code);
    }
}

#[derive(Eq)]
struct TreeNode {
    value: char,
    occurence: i64,
    childs: Vec<TreeNode>
}

impl TreeNode {
    pub fn new(p:(char, i64, Vec<TreeNode>)) -> TreeNode {
        TreeNode {
            value: p.0,
            occurence: p.1,
            childs: p.2
        }
    }
}

impl Ord for TreeNode {
    fn cmp(&self, other: &Self) -> Ordering {
        self.occurence.cmp(&other.occurence)
    }
}

impl PartialOrd for TreeNode {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl PartialEq for TreeNode {
    fn eq(&self, other: &Self) -> bool {
        self.occurence == other.occurence
    }
}

struct State {
    childs: Vec<State>,
    set: HashSet<usize>,
    position: HashSet<(usize, usize)>
}

impl State {
    fn add(&mut self, set: &HashSet<usize>, set_pos: &HashSet<(usize, usize)>) {
        for child in self.childs.iter_mut() {
            if child.set.is_superset(&set) {
                for i in set_pos {
                    child.position.remove(i);
                }
                child.add(&set, &set_pos);
                return;
            }
        }
        self.childs.push(State {
            childs: Vec::new(),
            set: set.clone(),
            position: set_pos.clone()
        });
    }

    fn search(&mut self, set: &HashSet<usize>) -> HashSet<(usize, usize)> {
        for child in self.childs.iter_mut() {
            if child.set.clone() == set.clone() {
                return child.position.clone();
            } else if child.set.is_superset(&set) {
                return child.search(&set);
            }
        }
        return self.position.clone();
    }

    fn print(&self, i: usize) {
        println!("{}: {:?} {:?}", i, self.set, self.position);
        for child in self.childs.iter() {
            child.print(i+1);
        }
        
    }
}

struct FactorOracle {
    states: Vec<Vec<usize>>,
    trans: HashMap<(Vec<usize>, char), Vec<usize>>,
    state_set_tree: State,
    occurence: HashMap<char, i64>
}

impl FactorOracle {
    pub fn new(tries: Trie) -> FactorOracle {
        let mut fa_states: Vec<Vec<usize>> = Vec::new();
        fa_states.push((0..tries.trans.len()).collect::<Vec<usize>>());
    //    let mut initial_pos: HashSet<(usize, usize)> = HashSet::new();
    //    for i in dist.iter() {
    //        for j in i {
    //            println!("{:?}", j);
    //            initial_pos.insert(*j);
    //        }
    //    }
        let mut state_set_tree = State {
            set: (0..tries.trans.len()).collect::<HashSet<usize>>(),
            childs: Vec::new(),
            position: HashSet::new()
    //        position: initial_pos
        };

        let mut fa_trans = HashMap::new();
        let mut occurence: HashMap<char, i64> = HashMap::new();

        let mut i = 0;
        while fa_states.len() > i {
            'outer: for t in tries.alphabet.iter() {
                let mut tmp = Vec::new();
                let mut tmp_pos = HashSet::new();
                for s in fa_states[i].iter() {
                    match tries.trans.get(&(*s, *t)) {
                        Some(x) => {
                            for t in tries.position[*x - 1].iter() {
                                tmp_pos.insert(*t);
                            }
                            tmp.push(*x);
                        },
                        None => {}
                    }
                }
                if tmp.len() > 0 {
                    let counter = occurence.entry(*t).or_insert(0);
                    *counter += 1;
                    tmp.sort();
                    'inner: for u in fa_states.iter() {
                        if tmp[0] == u[0] {
                            fa_trans.insert((fa_states[i].clone(), *t), u.clone());
                            continue 'outer;
                        }
                    }
                    fa_trans.insert((fa_states[i].clone(), *t), tmp.clone());
                    fa_states.push(tmp.clone());
                    state_set_tree.add(&tmp.into_iter().collect::<HashSet<usize>>(), &tmp_pos);
                }
            }
            i+=1;
        }

        FactorOracle {
            states: fa_states,
            trans: fa_trans,
            state_set_tree: state_set_tree,
            occurence: occurence
        }
    }
}

struct Trie {
    alphabet: HashSet<char>,
    trans: HashMap<(usize, char), usize>,
    position: Vec<Vec<(usize, usize)>>
}

impl Trie {
    pub fn new(s: Vec<Vec<char>>) -> Trie {
        let mut trie_trans = HashMap::new();
        let mut trie_alphabet = HashSet::new();
        let mut last = 0;
        let mut dist: Vec<Vec<(usize, usize)>> = Vec::new();
        for i in 0..s.len() {
            let mut crt = 0;
            for j in 0..s[i].len() {
                trie_alphabet.insert(s[i][j]);
                match trie_trans.get(&(crt, s[i][j])) {
                    Some(x) => {
                        crt=*x;
                        dist[crt-1].push((i, j));
                    },
                    None => {
                        dist.push(Vec::new());
                        dist[last].push((i, j));
                        last+=1;
                        trie_trans.insert((crt, s[i][j]), last);
                        crt=last;
                    }
                }
            }
        }

        Trie {
            alphabet: trie_alphabet,
            trans: trie_trans,
            position: dist
        }
    }
}

