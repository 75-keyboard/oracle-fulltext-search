use std::collections::{ HashMap, HashSet, BinaryHeap };
use std::cmp::{ Ordering, max, min };
use std::fs::read_to_string;
use std::fs::File;
use std::io::{self, BufRead, BufReader};
use bit_vec::BitVec;

fn main(){
    // トライ木からファクターオラクルを構築
//    let fa = FactorOracle::new();
 //   println!("{:?}", fa.trans);
   // println!("created FO--------------------------------");
//    println!("{:?}", fa.trans);
//    println!("{:?}", fa.order);

    //fa.state_set_tree.print(0);
    //let mut searchable: HashSet<usize> = HashSet::new();
    //searchable.insert(6);
    //println!("{:?}", fa.state_set_tree.search(&searchable));

    // ファクターオラクルの遷移のラベルの出現頻度からハフマン符号を構成
    let fa = FactorOracleOnline::new();
    //println!("{:?}", fa);
    create_index(fa);
}

fn create_index(fa: FactorOracleOnline) {
    let hc = HuffmanCode::new(fa.occurence);
    hc.print_code_info();
    let mut state_table: HashMap<usize, usize> = HashMap::new();

    let mut state_index = 0;
    let mut bv = BitVec::from_elem(min(100000000000, 32*(hc.code_info.len()*fa.state_num)), false);

    // ビット列に変換
    for i in (0..fa.state_num).rev() {
        //println!("{}", i);
        //println!("{:?}", fa.trans[i]);
        //println!("{:?}", state_table);
        for (j, k) in fa.trans[i].iter() {
//            println!("{:?}", i);
            match state_table.get(k) {
                Some(_) => {},
                None => {
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
                    match fa.trans.get(*k) {
                        Some(x) if x.len() != 0 => {
                            println!("{:?}", x);
                            let mut max_index = 0;
                            for j in x.iter() {
                                max_index = max(max_index, state_table[j.1]);
                            }
                            let width = decide_bits(max_index);
                            //print!(" width {:?} ", width);
                            for ii in (0..3).rev() {
                                if 1 << ii & width.1 != 0 {
                                    bv.set(state_index, true);
                                } else {
                                    bv.set(state_index, false);
                                }
                                state_index+=1;
                            }
                            for j in x.iter() {
                                //print!("{:?} {}", j, state_table[j.1]);
                                for ii in (0..width.0).rev() {
                                    if 1 << ii & state_table[j.1] != 0 {
                                        bv.set(state_index, true);
                                    } else {
                                        bv.set(state_index, false);
                                    }
                                    state_index+=1;
                                }
                            }
                        }, _ => {}
                    }
                }
            }
            //println!("");
        }
    }
    // println!("owa {}", state_table.len());
//    println!("{:?}", fa.trans);
    let init = state_index.clone();
    match fa.trans.get(0) {
        Some(x) => {
            //print!("{:?}", x);
            let mut max_index = 0;
            for j in x.iter() {
                max_index = max(max_index, state_table[j.1]);
            }
            let width = decide_bits(max_index);
            //print!(" width {:?} ", width);
            for ii in (0..3).rev() {
                if 1 << ii & width.1 != 0 {
                    bv.set(state_index, true);
                } else {
                    bv.set(state_index, false);
                }
                state_index+=1;
            }
            for j in x.iter() {
                //print!("{:?} {}", j, state_table[j.1]);
                for ii in (0..width.0).rev() {
                    if 1 << ii & state_table[j.1] != 0 {
                        bv.set(state_index, true);
                    } else {
                        bv.set(state_index, false);
                    }
                    state_index+=1;
                }
            }
            //println!("");
        }, None => {}
    }

    println!("index created");
    bv.truncate(state_index);
    println!("{:?}", bv);
    println!("{:?}", bv.to_bytes().len());
    println!("Bits used {} of {}", state_index, min(100000000000, 512*(hc.code_info.len()*fa.state_num)));
}

/**
 * 引数のビット数を計算
 */
fn count_bits(x: usize) -> usize {
    let mut i = 0;
    while x >> i > 0 {
        i += 1;
    }
    i
}

/**
 * 遷移のビット数を決定する
 */
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


/**
 * ハフマン符号
 */
struct HuffmanCode {
    code_info: HashMap<char, (usize, usize)>
}

impl HuffmanCode {
    /**
     * ハフマン木を構成し、ハフマン符号にする
     */
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
        HuffmanCode::dfs(&mut code_info, &root, 0, 0);
        HuffmanCode {
            code_info: code_info
        }
    }

    /**
     * 符号情報を出力
     */
    fn print_code_info(&self) {
        for (k, v) in self.code_info.iter() {
            println!("{}: {:0b} {}", k, v.0, v.1);
        }
    }

    /**
     * ハフマン木をたどる
     */
    fn dfs(code_info: &mut HashMap<char, (usize, usize)>, node: &TreeNode, code: usize, code_size: usize) {
        if !node.childs.is_empty() {
            HuffmanCode::dfs(code_info, &node.childs[0], code << 1, code_size+1);
            HuffmanCode::dfs(code_info, &node.childs[1], (code << 1) + 1, code_size+1);
        } else {
            code_info.insert(node.value, (code, code_size));
        }
    }
}

/**
 * ハフマン木の各ノード 頻度による大小関係をderive
 */
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

#[derive(Debug)]
struct FactorOracleOnline {
    state_num: usize,
    trans: Vec<HashMap<char, usize>>,
    supply_function: Vec<usize>,
    occurence: HashMap<char, i64>
}

impl FactorOracleOnline {
    /**
     * Allauzenのオンラインファクターオラクル構成法
     */
    pub fn new() -> FactorOracleOnline {
        let p :Vec<char> = "abbbaab".chars().collect();
        //let p :Vec<char> = read_to_string("data/test.txt").unwrap().chars().collect::<Vec<char>>();

        let mut oracle = FactorOracleOnline::init();
        let l = p.len();

        for i in 0..l {
            //println!("{} / {}", i, l);
            oracle.add_letter(p[i].clone());
        }

        oracle
    }

    /**
     * 初期化
     */
    fn init() -> FactorOracleOnline {
        let mut supply_function :Vec<usize> = Vec::new();
        supply_function.push(1_000_000_000_000);        
        let mut trans: Vec<HashMap<char, usize>>= Vec::new();
        trans.push(HashMap::new());
        FactorOracleOnline { 
            state_num: 0,
            trans: trans,
            supply_function,
            occurence: HashMap::new()
        }
    }

    /**
     * 1文字追加関数
     */
    fn add_letter(&mut self, sigma: char) {
        let i = self.state_num.clone();
        self.state_num += 1;
        self.trans.push(HashMap::new());
        self.trans[i].insert(sigma, i+1);

        let counter = self.occurence.entry(sigma).or_insert(0);
        *counter += 1;

        let mut k = self.supply_function[i];
        
        while k != 1_000_000_000_000 {
            if let Some(_) = self.trans[k].get(&sigma) { break }
            self.trans[k].insert(sigma, i+1);
            *counter += 1;
            k = self.supply_function[k];
        }

        if k == 1_000_000_000_000 { 
            self.supply_function.push(0);
        } else {
            if let Some(x) = self.trans[k].get(&sigma) { 
                self.supply_function.push(x.clone());
            }
        }
    }
}
