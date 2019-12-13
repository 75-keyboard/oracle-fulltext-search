use std::collections::{ HashMap, HashSet, BinaryHeap };
use std::cmp::{ Ordering, max, min };
use std::fs::read_to_string;
use std::fs::File;
use std::io::{self, BufRead, BufReader};
use bit_vec::BitVec;

fn main(){
    let tries = Trie::new();
//    println!("{:?}", tries.alphabet);
    println!("created trie--------------------------------");

    let fa = FactorOracle::new();
    println!("{:?}", fa.states);
    println!("created FO--------------------------------");
//    println!("{:?}", fa.trans);
//    println!("{:?}", fa.order);

    //fa.state_set_tree.print(0);
    //let mut searchable: HashSet<usize> = HashSet::new();
    //searchable.insert(6);
    //println!("{:?}", fa.state_set_tree.search(&searchable));
    let hc = HuffmanCode::new(fa.occurence);
    hc.print_code_info();
    let mut state_table: HashMap<usize, usize> = HashMap::new();

    let mut state_index = 0;
    let mut bv = BitVec::from_elem(min(100000000000, 32*(hc.code_info.len()*fa.states.len())), false);

    println!("{:?}", fa.order);
    // convert to binary
    for i in fa.order.iter() {
        for (j, k) in fa.trans[i].iter() {
//            println!("{:?}", i);
            match state_table.get(k) {
                Some(_) => {},
                None => {
                    print!("{} {}", j, *k);
                    state_table.insert(*k, state_index);
                    let trans_code = hc.code_info[j];
                    // 状態に入ってくる記号を入れてる
                    for ii in (0..trans_code.1).rev() {
                        if 1 << ii & trans_code.0 != 0 {
                            bv.set(state_index, true);
                        } else {
                            bv.set(state_index, false);
                        }
                        state_index+=1;
                    }
                    match fa.trans.get(k) {
                        Some(x) => {
                            let mut max_index = 0;
                            for j in x.iter() {
                                max_index = max(max_index, state_table[j.1]);
                            }
                            let width = decide_bits(max_index);
                            print!(" width {:?} ", width);
                            for ii in (0..3).rev() {
                                if 1 << ii & width.1 != 0 {
                                    bv.set(state_index, true);
                                } else {
                                    bv.set(state_index, false);
                                }
                                state_index+=1;
                            }
                            for j in x.iter() {
                                print!("{:?} {}", j, state_table[j.1]);
                                for ii in (0..width.0).rev() {
                                    if 1 << ii & state_table[j.1] != 0 {
                                        bv.set(state_index, true);
                                    } else {
                                        bv.set(state_index, false);
                                    }
                                    state_index+=1;
                                }
                            }
                        }, None => {}
                    }
                }
            }
            println!("");
        }
    }
    // println!("owa {}", state_table.len());
//    println!("{:?}", fa.trans);
    let init = state_index.clone();
    match fa.trans.get(&0) {
        Some(x) => {
            print!("{:?}", x);
            let mut max_index = 0;
            for j in x.iter() {
                max_index = max(max_index, state_table[j.1]);
            }
            let width = decide_bits(max_index);
            print!(" width {:?} ", width);
            for ii in (0..3).rev() {
                if 1 << ii & width.1 != 0 {
                    bv.set(state_index, true);
                } else {
                    bv.set(state_index, false);
                }
                state_index+=1;
            }
            for j in x.iter() {
                print!("{:?} {}", j, state_table[j.1]);
                for ii in (0..width.0).rev() {
                    if 1 << ii & state_table[j.1] != 0 {
                        bv.set(state_index, true);
                    } else {
                        bv.set(state_index, false);
                    }
                    state_index+=1;
                }
            }
            println!("");
        }, None => {}
    }

    println!("index created");
    println!("Bits used {} of {}", state_index, min(100000000000, 512*(hc.code_info.len()*fa.states.len())));
    bv.truncate(state_index);
    println!("{:?}", bv);
    println!("{:?}", bv.to_bytes().len());
}

fn count_bits(x: usize) -> usize {
    let mut i = 0;
    while x >> i > 0 {
        i += 1;
    }
    i
}

fn decide_bits(x: usize) -> (usize, usize) {
    let mut i = 4;
    let mut cnt = 0;
    let j = count_bits(x);
    while j > i {
        i += 4;
        cnt+=1;
    }
    (i, cnt)
}

struct HuffmanCode {
    code_info: HashMap<char, (usize, usize)>
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
        let mut code_info: HashMap<char, (usize, usize)> = HashMap::new();
        dfs(&mut code_info, &root, 0, 0);
        HuffmanCode {
            code_info: code_info
        }
    }

    fn print_code_info(&self) {
        for (k, v) in self.code_info.iter() {
            println!("{}: {:0b} {}", k, v.0, v.1);
        }
    }
}

fn dfs(code_info: &mut HashMap<char, (usize, usize)>, node: &TreeNode, code: usize, code_size: usize) {
    if !node.childs.is_empty() {
        dfs(code_info, &node.childs[0], code << 1, code_size+1);
        dfs(code_info, &node.childs[1], (code << 1) + 1, code_size+1);
    } else {
        code_info.insert(node.value, (code, code_size));
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
    trans: HashMap<usize, HashMap<char, usize>>,
    state_set_tree: State,
    occurence: HashMap<char, i64>,
    order: Vec<usize>,
}

impl FactorOracle {
    fn create() -> (FactorOracle, HashMap<usize, HashSet<usize>>, HashMap<usize, HashSet<usize>>) {
        let mut tries = Trie::new();
        let mut fa_states: Vec<Vec<usize>> = Vec::new();
        fa_states.push((0..tries.trans.len()).collect());
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

        let mut fa_trans: HashMap<usize, HashMap<char, usize>> = HashMap::new();
        let mut fa_trans_in: HashMap<usize, HashSet<usize>> = HashMap::new();
        let mut fa_trans_out: HashMap<usize, HashSet<usize>> = HashMap::new();
        let mut occurence: HashMap<char, i64> = HashMap::new();
        let mut state_table: HashMap<usize, usize> = HashMap::new();

        let mut i = 0;
        while fa_states.len() > i {
println!("{} {} {}", i, fa_states[i].len(), tries.alphabet.len());
            'outer: for t in tries.alphabet.iter() {
                let mut tmp: Vec<usize> = Vec::new();
                let mut tmp_pos: HashSet<(usize, usize)> = HashSet::new();
                for s in fa_states[i].iter() {
                    match tries.trans.get(&(*s, *t)) {
                        Some(x) => {
                            tmp.push(*x);
                        }, None => {}
                    }
                }
                if tmp.len() > 0 {
                    let counter = occurence.entry(*t).or_insert(0);
                    *counter += 1;
                    tmp.sort();
                    match state_table.get(&tmp[0]) {
                        Some(&u) => {
                            match fa_trans.get_mut(&i) {
                                Some (x) => {
                                    x.insert(*t, u);
                                },
                                None => {
                                    let mut h: HashMap<char, usize> = HashMap::new();
                                    h.insert(*t, u);
                                    fa_trans.insert(i, h);
                                }
                            }
                            match fa_trans_in.get_mut(&u) {
                                Some (x) => {
                                    x.insert(i);
                                },
                                None => {
                                    let mut h: HashSet<usize> = HashSet::new();
                                    h.insert(i);
                                    fa_trans_in.insert(u, h);
                                }
                            }
                            match fa_trans_out.get_mut(&i) {
                                Some (x) => {
                                    x.insert(u);
                                },
                                None => {
                                    let mut h: HashSet<usize> = HashSet::new();
                                    h.insert(u);
                                    fa_trans_out.insert(i, h);
                                }
                            }
                            continue 'outer;
                        },
                        None => {
                            for i in tmp.clone().iter() {
                                for j in tries.position[*i-1].iter() {
                                    tmp_pos.insert(*j);
                                }
                            }
                            match fa_trans.get_mut(&i) {
                                Some (x) => {
                                    x.insert(*t, fa_states.len());
                                },
                                None => {
                                    let mut h: HashMap<char, usize> = HashMap::new();
                                    h.insert(*t, fa_states.len());
                                    fa_trans.insert(i, h);
                                }
                            }
                            match fa_trans_in.get_mut(&fa_states.len()) {
                                Some (x) => {
                                    x.insert(i);
                                },
                                None => {
                                    let mut h: HashSet<usize> = HashSet::new();
                                    h.insert(i);
                                    fa_trans_in.insert(fa_states.len(), h);
                                }
                            }
                            match fa_trans_out.get_mut(&i) {
                                Some (x) => {
                                    x.insert(fa_states.len());
                                },
                                None => {
                                    let mut h: HashSet<usize> = HashSet::new();
                                    h.insert(fa_states.len());
                                    fa_trans_out.insert(i, h);
                                }
                            }  
                            state_table.insert(tmp[0], fa_states.len());
                            fa_states.push(tmp.clone());
                            state_set_tree.add(&tmp.into_iter().collect::<HashSet<usize>>(), &tmp_pos);
                        }
                    }
                }
            }
            i+=1;
        }
        // println!("{:?}", fa_states);
        // println!("IN: {:?}", fa_trans_in);
        for i in 0..fa_states.len() {
            if fa_trans_out.get(&i) == None {
                fa_trans_out.insert(i, HashSet::new());
            } 
        }
        // println!("OUT: {:?}", fa_trans_out);
        (
            FactorOracle {
                states: fa_states,
                trans: fa_trans,
                state_set_tree: state_set_tree,
                occurence: occurence,
                order: Vec::new(),
            }, fa_trans_in, fa_trans_out
        )
        
    } 
    pub fn new() -> FactorOracle {
        let (mut fo, mut fa_trans_in, mut fa_trans_out) = FactorOracle::create();
        let mut ii = 0;
        for j in fa_trans_out.clone().iter_mut() {
            if j.1.len() == 0 {
                fo.order.push(*j.0);
                fa_trans_out.remove(j.0);
                ii += 1;
            }
        }
        let l = fa_trans_out.len();
        let mut i = 0;
        while fa_trans_out.len() > 0 {
            match fa_trans_in.get_mut(&fo.order[i]) {
                Some(x) => {
                    for j in x.iter() {
                        if fa_trans_out.entry(*j).or_default().remove(&fo.order[i]) && fa_trans_out.entry(*j).or_default().len() == 0 {
                            fa_trans_out.remove(j);
                            fo.order.push(*j);
                        }
                    }
                }, None => {}
            }
            i+=1;
            println!("{}, {}", i, l);
        }
        fo.order = fo.order[ii..].to_vec();
        fo
    }
}

struct Trie {
    alphabet: Vec<char>,
    trans: HashMap<(usize, char), usize>,
    position: Vec<Vec<(usize, usize)>>
}

impl Trie {
    pub fn new() -> Trie {
        let mut s: Vec<Vec<char>> = Vec::new();
    //    s.push(Bytes::from(&b"abcba"[..]));
    //    s.push(Bytes::from(&b"abbac"[..]));
        s.push(read_to_string("data/manual.txt").unwrap().chars().collect::<Vec<char>>());
        // for result in BufReader::new(File::open("data/test.txt").unwrap()).lines() {
        //     s.push(result.unwrap().chars().collect());
        // }
    //    println!("{:?}", s);
        //s.push("abcba".chars().collect());
        //s.push("abbac".chars().collect());
    
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
        let mut trie_alphabet = trie_alphabet.into_iter().collect::<Vec<char>>();
        trie_alphabet.sort();

        Trie {
            alphabet: trie_alphabet,
            trans: trie_trans,
            position: dist
        }
    }
}

