use std::collections::{ BTreeMap, HashMap, HashSet, BinaryHeap };
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

    let mut s: Vec<Vec<char>> = Vec::new();
    //for i in 0..100 {
    //    s.push(read_to_string(format!("data/txt/man{}.txt", i)).unwrap().chars().collect());
    //}
    //s.push(read_to_string("data/manual.txt").unwrap().chars().collect::<Vec<char>>());
    // for result in BufReader::new(File::open("data/test.txt").unwrap()).lines() {
    //     s.push(result.unwrap().chars().collect());
    // }
//    println!("{:?}", s);
    
    // ファクターオラクルの遷移のラベルの出現頻度からハフマン符号を構成
    let fa = FactorOracleOnline::new({
        let mut s = Vec::new();
        for i in 0..10 {
            s.push(read_to_string(format!("data/txt/man{}.txt", i)).unwrap().chars().collect());
        }
        println!("Files loaded.");

        s
    });
    //println!("{:?}", fa);

    //println!("{:?}", fa.search("abbac".to_string()));
    //println!("{:?}", fa.search("html".to_string()));
    //println!("{:?}", fa.search("test".to_string()));
    loop {
        let s = {
            let mut s = String::new(); // バッファを確保
            std::io::stdin().read_line(&mut s).unwrap(); // 一行読む。失敗を無視
            s.trim_right().to_owned() // 改行コードが末尾にくっついてくるので削る
        };

        // println!("{:?}", fa.search(s));
        println!("{:?}", fa.search(s).len());
    }
    //create_index(fa);
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

/**
 * トライ木
 */
struct Trie {
    alphabet: Vec<char>,
    trans: HashMap<(usize, char), usize>,
    //trans: BTreeMap<(usize, char), usize>,
    position: Vec<Vec<(usize, usize)>>,
    txt: Vec<Vec<char>>,
}

impl Trie {
    /**
     * 文字列からトライ木を構築
     */
    pub fn new(s: Vec<Vec<char>>) -> Trie {
        let mut trie_trans = HashMap::new();
        //let mut trie_trans = BTreeMap::new();
        //let mut reverse_trans: BTreeMap<usize, (char, usize)> = BTreeMap::new();
        let mut trie_alphabet = HashSet::new();
        //let mut finals = Vec::new();
        let mut last = 0;
        let mut dist: Vec<Vec<(usize, usize)>> = Vec::new();
        dist.push(Vec::new());
        let mut summ = 0;
        for i in 0..s.len() {
            dist[i].push((i, 0));
            let mut crt = 0;
            summ += s[i].len();
            for j in 0..s[i].len() {
                trie_alphabet.insert(s[i][j]);
                match trie_trans.get(&(crt, s[i][j])) {
                    Some(x) => {
                        crt=*x;
                        dist[crt-1].push((i, j+1));
                        //if j == s[i].len()-1 { finals.push(crt-1) }
                    },
                    None => {
                        dist.push(Vec::new());
                        dist[last+1].push((i, j+1));
                        last+=1;
                        trie_trans.insert((crt, s[i][j]), last);
                        //reverse_trans.insert(last, (s[i][j], crt));
                        crt=last;
                        //if j == s[i].len()-1 { finals.push(crt) }
                    }
                }
            }
        }
        println!("SUMMARY {}", summ);

        //println!("finals {:?}", finals);
        //let mut next = finals.clone();
        //let fisrt = next[0].clone();
        //let mut mm: Vec<HashMap<char, Vec<usize>>> = Vec::new();
        //'a: while true {
        //    let mut hm: HashMap<char, Vec<usize>> = HashMap::new();
        //    for i in 0..next.len() {
        //        println!("aa {:?}", next);
        //        match reverse_trans.get(&next[i]) {
        //            Some(x) => { 
        //                next[i]=x.1; 
        //                match hm.get_mut(&x.0) {
        //                    Some(y) => {
        //                        y.push(x.1);
        //                    },
        //                    None => {
        //                        hm.insert(x.0, vec![x.1]);
        //                    }
        //                }          
        //            }, None => { break 'a; }
        //        };
        //    }
        //    mm.push(hm);
        //}
        //println!("{:?}", mm);

        let mut trie_alphabet = trie_alphabet.into_iter().collect::<Vec<char>>();
        trie_alphabet.sort();
        println!("Trie constructed.");

        Trie {
            alphabet: trie_alphabet,
            trans: trie_trans,
            position: dist,
            txt: s,
        }
    }
}

#[derive(Debug)]
struct FactorOracleOnline {
    state_num: usize,
    trans: Vec<HashMap<char, usize>>,
    supply_function: Vec<usize>,
    suffix_link_tree: Vec<Vec<usize>>,
    searching_info: (Vec<usize>, Vec<usize>),
    sp_inverse: Vec<(usize, usize)>,
    position: Vec<Vec<(usize, usize)>>,
    occurence: HashMap<char, i64>,
    txt: Vec<Vec<char>>,
}

impl FactorOracleOnline {
    /**
     * Allauzenのオンラインファクターオラクル構成法
     */
    pub fn new(s: Vec<Vec<char>>) -> FactorOracleOnline {
        //let p :Vec<char> = "abbbaab".chars().collect();
        let trie = Trie::new(s);

        let mut oracle = FactorOracleOnline::init(trie.trans.len()+1, trie.position.clone(), trie.txt);

        let mut v: Vec<_> = trie.trans.into_iter().collect();
        v.sort_by(|x,y| x.0.cmp(&y.0));

        for ((src, sigma), dst) in v {
            //println!("{} {} {}", src, sigma, dst);
            oracle.add_state(src, sigma, dst);
        }

        //for src in 0..trie.position.len() {
        //    for sigma in &trie.alphabet {
        //        match trie.trans.get(&(src, *sigma)) {
        //            Some(dst) => {
        //                oracle.add_state(src, *sigma, *dst);
        //            }, None => {}
        //        }
        //    }
        //}

        oracle.organize_seaching_info();

        println!("Oracle constructed");
        //println!("{:?}", oracle);
        oracle
    }

    fn organize_seaching_info(&mut self) {
        let searching_info = self.re_order(0, Vec::new(), vec![0; self.supply_function.len()], vec![(0, 0); self.supply_function.len()]);


        self.searching_info = (searching_info.0, searching_info.1);
        self.sp_inverse = searching_info.2;
    }

    /**
     * 初期化
     */
    fn init(len: usize, position: Vec<Vec<(usize, usize)>>, txt: Vec<Vec<char>>) -> FactorOracleOnline {
        let mut supply_function :Vec<usize> = vec![0; len];
        supply_function[0] = 1_000_000_000_000;      
        let trans: Vec<HashMap<char, usize>>= vec![HashMap::new(); len];
        FactorOracleOnline { 
            state_num: 0,
            trans: trans,
            supply_function,
            suffix_link_tree: vec![Vec::new(); len],
            searching_info: (Vec::new(), Vec::new()),
            sp_inverse: Vec::new(),
            position: position,
            occurence: HashMap::new(),
            txt: txt,
        }
    }

    /**
     * 1文字追加関数
     */
    fn add_state(&mut self, src: usize, sigma: char, dst: usize) {
        let q = src.clone();
        let i = dst.clone();
        self.state_num += 1;
        self.trans[q].insert(sigma, i);

        let counter = self.occurence.entry(sigma).or_insert(0);
        *counter += 1;

        let mut k = self.supply_function[q];
        while k != 1_000_000_000_000 {
            if let Some(_) = self.trans[k].get(&sigma) { break }
            self.trans[k].insert(sigma, i);
            *counter += 1;
            k = self.supply_function[k];
        }

        if k == 1_000_000_000_000 { 
            self.supply_function[i] = 0;
            self.suffix_link_tree[0].push(i);
        } else {
            if let Some(&x) = self.trans[k].get(&sigma) { 
                self.supply_function[i] = x.clone();
                self.suffix_link_tree[x].push(i);
            }
        }

    }

    fn re_order(&self, idx: usize, mut x: Vec<usize>, mut y: Vec<usize>, mut z: Vec<(usize, usize)>) -> (Vec<usize>, Vec<usize>, Vec<(usize, usize)>) {
        //println!("{} {}", idx, x.len());
        let a= x.len();
        y[idx] = x.len();
        x.push(idx);
        for i in &self.suffix_link_tree[idx] {
            let xy = self.re_order(*i, x, y, z);
            x = xy.0; y = xy.1; z = xy.2;
        }
        z[a] = (a, x.len()-1);
        (x, y, z)
    }

    fn search(&self, w: String) -> Vec<(usize, usize)> {
        let mut endpos: Vec<(usize, usize)> = Vec::new();
        let mut current_state = 0;
        for i in w.chars() {
            match self.trans[current_state].get(&i) {
                Some(x) => {
                    current_state = *x;
                }, None => { current_state = 1_000_000_000_000; break; }
            }
        }

        // 受理した状態に対応する位置情報を確認
        if current_state != 1_000_000_000_000 { 
            println!("Accepted {:?} by {} is {}", w, current_state, self.searching_info.1[current_state]);
            for position in self.position[current_state].clone() {
                //println!("{:?}", position);
                let ww: Vec<char> = w.chars().collect();
                for i in 0 .. ww.len() {
                    // println!("{} == {} {} + {} - {}", ww[i ], self.txt[position.0][position.1 + i - ww.len()], position.1, i ,ww.len());
                    if ww[i] != self.txt[position.0][position.1 + i - ww.len()] {
                        break;
                    }

                    if i == ww.len() - 1 { endpos.push(position); }
                }
            }

            let mut sp_inverse = self.sp_inverse[self.searching_info.1[current_state]];
            //println!("SP_INVERSE {:?}", sp_inverse);
            let mut s_inverse = Vec::new();
            if sp_inverse.0 != sp_inverse.1 {
                let end = sp_inverse.1;
                sp_inverse = self.sp_inverse[sp_inverse.0+1];
                loop {
                    //println!("{:?}", sp_inverse.0);
                    s_inverse.push(sp_inverse.0);
                    if sp_inverse.1 == end { break; }
                    sp_inverse = self.sp_inverse[sp_inverse.1+1];
                }
            }
            println!("S_INVERSE {:?}", s_inverse.len());

            let ww: Vec<char> = w.chars().collect();
            for j in s_inverse {
                for position in self.position[self.searching_info.0[j]].clone() {
                    //println!("{:?}", position);
                    for i in 0 .. ww.len() {
                        //println!("{} + {} - {}", position.1, i ,ww.len());
                        if ww[i] != self.txt[position.0][position.1 + i - ww.len()] {
                            break;
                        }

                        if i == ww.len() - 1 { 
                            sp_inverse = self.sp_inverse[j];
                            //println!("SP_INVERSE {:?}", sp_inverse);
                            for k in sp_inverse.0 .. sp_inverse.1+1 {
                                //println!("koko {} {}", k, self.searching_info.0[k]);
                                //endpos = [endpos.clone(), self.position[self.searching_info.0[k]].clone()].concat(); 
                                for kk in self.position[self.searching_info.0[k]].clone() {
                                    endpos.push(kk);
                                }
                            }
                        }
                    }
                }
            }
        }

        endpos
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_search() {
        // ファクターオラクルの遷移のラベルの出現頻度からハフマン符号を構成
        let fa = FactorOracleOnline::new(vec!["abbac".chars().collect(), "aaaac".chars().collect()]);

        //println!("{:?}", fa.search("abbac".to_string()));
        assert_eq!(vec![(1, 2), (1, 3), (1, 4)], fa.search("aa".to_string()));
        assert_eq!(vec![(0, 2), (0, 3)], fa.search("b".to_string()));
        assert_eq!(vec![(0, 5), (1, 5)], fa.search("ac".to_string()));
    }
}
