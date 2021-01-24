use std::collections::{ HashMap, HashSet, BinaryHeap };
use std::cmp::{ Ordering, max, min };
use std::fs::read_to_string;
use std::fs::File;
use std::io::{self, BufRead, BufReader};
use bit_vec::BitVec;

use std::time::{Instant, Duration};

fn main(){
    //exec1(100000);
    //exec1(200000);
    //exec1(300000);
    //exec1(400000);
    //exec1(500000);
    exec2(100000);
    //exec3();
}

fn exec3() {
    let oracle = FactorOracle::small({
        let mut s: Vec<Vec<char>> = Vec::new();
        let mut sum = 0;
        for i in 0..10000 {
            //s.push(read_to_string(format!("data/txt/man{}.txt", i)).unwrap().chars().collect());
            //s.push(read_to_string(format!("data/maildata4/{}.txt", i)).unwrap().chars().collect());
            for result in BufReader::new(File::open(format!("data/maildata4/{}.txt", i)).unwrap()).lines() {
                let r = result.unwrap();
                sum += r.len();
                s.push(r.chars().collect());
            }
        }
        //s.push("abbac".chars().collect());
        //s.push("aaaac".chars().collect());
        s
    });

    println!("{:?}", std::mem::size_of_val(&oracle));
}

fn exec2(idx: usize) {
    let mut max = 0;
    let mut min = 1_000_000_000;
    let mut num_words = 0;
    let mut num = 0;
    // ファクターオラクルの遷移のラベルの出現頻度からハフマン符号を構成
    let start = Instant::now();
    let mut ss: HashMap<usize, Vec<usize>> = HashMap::new();
    let oracle = FactorOracle::small({
        let mut s: Vec<Vec<char>> = Vec::new();
        for i in 0..idx {
            //s.push(read_to_string(format!("data/txt/man{}.txt", i)).unwrap().chars().collect());
            //s.push(read_to_string(format!("data/maildata4/{}.txt", i)).unwrap().chars().collect());
            for result in BufReader::new(File::open(format!("data/maildata4/{}.txt", i)).unwrap()).lines() {
                let r = result.unwrap();
                num_words += 1;
                num += r.len();
                max = std::cmp::max(max, r.len());
                min = std::cmp::min(min, r.len());
                let tmp = ss.entry(r.len()).or_insert(Vec::new());
                tmp.push(s.len());
                s.push(r.chars().collect());
            }
        }
        //s.push("abbac".chars().collect());
        //s.push("aaaac".chars().collect());
        s
    });
    let end = start.elapsed();
    println!("{}, {}, {}, {}.{:03}", idx, num_words, num, end.as_secs(), end.subsec_nanos() / 1_000_000);

    println!("パターン長, 検索時間(1000回平均)");
    use rand::Rng;
    for x in 1..max {
        let mut sum = 0;
        for _ in 0..1000 {
            let mut rng = rand::thread_rng(); // デフォルトの乱数生成器を初期化します
            let mut i;
            if x < min {
                i = rng.gen::<usize>() % (max - min);           // genはRng traitに定義されている
                i += min;
            } else {
                i = rng.gen::<usize>() % (max - x);           // genはRng traitに定義されている
                i += x;
            }

            let j: usize = rng.gen::<usize>() % ss[&i].len();           // genはRng traitに定義されている
            let s = oracle.txt[ss[&i][j]].clone()[0..x].iter().collect::<String>().chars().collect::<Vec<char>>();
            //println!("{}, {}", x, s);
            let positions = oracle.get_candidate(s.clone());

            let start = Instant::now();
            let result = oracle.exec_search(s, positions);
            let end = start.elapsed();
            sum += end.as_micros();
        }
        println!("{}, {}", x, sum/1000);
    }
}

fn exec1(idx: usize) {
    let mut sum = 0;    
    let mut nw = 0;
    let mut n = 0;

    for _ in 0..10 {
        let mut num_words = 0;
        let mut num = 0;

        let start = Instant::now();
        let oracle = FactorOracle::small({
            let mut s: Vec<Vec<char>> = Vec::new();
            for i in 0..idx {
                //s.push(read_to_string(format!("data/txt/man{}.txt", i)).unwrap().chars().collect());
                //s.push(read_to_string(format!("data/maildata4/{}.txt", i)).unwrap().chars().collect());
                for result in BufReader::new(File::open(format!("data/maildata4/{}.txt", i)).unwrap()).lines() {
                    let r = result.unwrap();
                    num_words += 1;
                    num += r.len();
                    s.push(r.chars().collect());
                }
            }
            //s.push("abbac".chars().collect());
            //s.push("aaaac".chars().collect());
            s
        });
        let end = start.elapsed();
        sum += end.as_micros();
        n += num;
        nw += num_words;
    }
    println!("{}, {}, {}, {},", idx, nw/10, n/10, sum/10);

    //let fa = FullTextFactorOracle::new_both({
    //    let mut s: Vec<Vec<char>> = Vec::new();
    //    //for i in 0..1000 {
    //    //    s.push(read_to_string(format!("data/txt/man{}.txt", i)).unwrap().chars().collect());
    //    //}
    //    //s.push("abbac".chars().collect());
    //    //s.push("aaaac".chars().collect());
    //    //
    //    println!("Files loaded.");

    //    s
    //});
    //println!("{:?}", fa);

    //println!("{:?}", fa.search("abbac".to_string()));
    //println!("{:?}", fa.search("html".to_string()).len());
    //println!("{:?}", fa.search("test".to_string()));

//    use rand::Rng;
//    let mut sum = 0;
//    for _ in 0..1000 {
//        //let s = {
//        //    let mut s = String::new(); // バッファを確保
//        //    std::io::stdin().read_line(&mut s).unwrap(); // 一行読む。失敗を無視
//        //    s.trim_end().to_owned() // 改行コードが末尾にくっついてくるので削る
//        //};
//        use rand::Rng;
//        let mut rng = rand::thread_rng(); // デフォルトの乱数生成器を初期化します
//        let i: usize = rng.gen();           // genはRng traitに定義されている
//        let i: usize = i % num_words;           // genはRng traitに定義されている
//        let s = oracle.txt[i as usize].clone();
//
//        let start = Instant::now();
//        let result = oracle.search(s.iter().collect::<String>());
//        let end = start.elapsed();
//        sum += end.as_micros();
//        //println!("{:?}", result);
//        //println!("{:?} {}.{:03}", result.len(), end.as_secs(), end.subsec_nanos() / 1_000_000);
//    }
//    println!("{}", sum/1000);
}

fn create_index(s: Vec<Vec<char>>) {
    // トライ木からファクターオラクルを構築
    let fa = FactorOracle::create(s).0;
    println!("{:?}", fa.states);
    println!("created FO--------------------------------");
//    println!("{:?}", fa.trans);
//    println!("{:?}", fa.order);

    //fa.state_set_tree.print(0);
    //let mut searchable: HashSet<usize> = HashSet::new();
    //searchable.insert(6);
    //println!("{:?}", fa.state_set_tree.search(&searchable));

    // ファクターオラクルの遷移のラベルの出現頻度からハフマン符号を構成
    let hc = HuffmanCode::new(fa.occurence);
    hc.print_code_info();
    let mut state_table: HashMap<usize, usize> = HashMap::new();

    let mut state_index = 0;
    let mut bv = BitVec::from_elem(min(100000000000, 32*(hc.code_info.len()*fa.states.len())), false);

    println!("{:?}", fa.order);
    // ビット列に変換
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
 * 状態集合木
 */
#[derive(Clone)]
struct State {
    childs: Vec<State>,
    set: HashSet<usize>,
    position: HashSet<(usize, usize)>
}

impl State {
    /**
     * ノードを追加
     */
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

    /**
     * ノードの検索
     */
    fn search(self, set: &HashSet<usize>) -> State {
        for child in self.childs.iter() {
            if child.set.clone() == set.clone() {
                return child.clone();
            } else if child.set.is_superset(&set) {
                return child.clone().search(&set);
            }
        }
        return self.clone();
    }

    /**
     * 出力
     */
    fn print(&self, i: usize) {
        println!("{}: {:?} {:?}", i, self.set, self.position);
        for child in self.childs.iter() {
            child.print(i+1);
        }
        
    }

    fn solve(&self) -> HashSet<(usize, usize)> {
        let mut hs = HashSet::new();
        self.down(&mut hs);
        hs
    }

    fn down(&self, hs: &mut HashSet<(usize, usize)>) {
        //println!("{:?}", self.set);
        for h in self.position.clone() {
            hs.insert(h);
        }
        for child in self.childs.clone() {
            child.down(hs);
        }
    }
}

/**
 * ファクターオラクル
 */
struct FactorOracle {
    states: Vec<Vec<usize>>,
    trans: HashMap<usize, HashMap<char, usize>>,
    state_set_tree: State,
    occurence: HashMap<char, i64>,
    order: Vec<usize>,
    txt: Vec<Vec<char>>
}

impl FactorOracle {
    fn exec_search(&self, p: Vec<char>, positions: HashSet<(usize, usize)>) -> Vec<(usize, usize)> {
        let mut output = Vec::new();
        for position in positions {
            if p.len() > position.1+1 { continue; }
            //println!("{:?}", position);
            for i in 0 .. p.len() {
                //println!("{} + {} - {}", position.1, i ,p.len());
                if p[i] != self.txt[position.0][position.1+1 + i - p.len()] {
                    break;
                } else if i == p.len() - 1 { 
                    output.push(position.clone());
                }
            }
        }

        output
    }

    fn get_candidate(&self, p: Vec<char>) -> HashSet<(usize, usize)> {
        let mut crt = 0;
        for i in 0..p.len() {
            crt = if let Some(x) = self.trans.get(&crt) {
                if let Some(&y) = x.get(&p[i]) {
                    y
                } else { return HashSet::new(); }
            } else { return HashSet::new(); };
        }

        let mut end = HashSet::new();
        for i in self.states[crt].clone() {
            end.insert(i);
        }
        
        let sst = self.state_set_tree.clone().search(&end);
        //println!("posi {:?}", sst.solve());
        sst.solve()
    }

    fn search(&self, p: Vec<char>) -> Vec<(usize, usize)> {
        let mut output = Vec::new();

        let mut crt = 0;
        for i in 0..p.len() {
            crt = if let Some(x) = self.trans.get(&crt) {
                if let Some(&y) = x.get(&p[i]) {
                    y
                } else { return Vec::new(); }
            } else { return Vec::new(); };
        }

        let mut end = HashSet::new();
        for i in self.states[crt].clone() {
            end.insert(i);
        }
        
        let sst = self.state_set_tree.clone().search(&end);
        //println!("posi {:?}", sst.solve());
        let positions = sst.solve();

        for position in positions {
            if p.len() > position.1+1 { continue; }
            //println!("{:?}", position);
            for i in 0 .. p.len() {
                //println!("{} + {} - {}", position.1, i ,p.len());
                if p[i] != self.txt[position.0][position.1+1 + i - p.len()] {
                    break;
                } else if i == p.len() - 1 { 
                    output.push(position.clone());
                }
            }
        }

        output
    }


    /**
     * 部分集合構成法により、トライ木からファクターオラクルを構築
     */
    fn small(s: Vec<Vec<char>>) -> FactorOracle {
        let mut tries = Trie::new(s);
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
        let mut occurence: HashMap<char, i64> = HashMap::new();
        let mut state_table: HashMap<usize, usize> = HashMap::new();

        let mut i = 0;
        while fa_states.len() > i {
//println!("{} {} {}", i, fa_states[i].len(), tries.alphabet.len());
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
        // println!("OUT: {:?}", fa_trans_out);
        (
            FactorOracle {
                states: fa_states,
                trans: fa_trans,
                state_set_tree: state_set_tree,
                occurence: occurence,
                order: Vec::new(),
                txt: tries.txt,
            }
        )
        
    } 

    /**
     * 部分集合構成法により、トライ木からファクターオラクルを構築
     */
    fn create(s: Vec<Vec<char>>) -> (FactorOracle, HashMap<usize, HashSet<usize>>, HashMap<usize, HashSet<usize>>) {
        let mut tries = Trie::new(s);
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
//println!("{} {} {}", i, fa_states[i].len(), tries.alphabet.len());
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
                txt: tries.txt
            }, fa_trans_in, fa_trans_out
        )
        
    } 

    /**
     * ファクターオラクルを構成し、ビット列にするために逆順のトポロジカルオーダーを記録
     */
    pub fn new(s: Vec<Vec<char>>) -> FactorOracle {
        let (mut fo, mut fa_trans_in, mut fa_trans_out) = FactorOracle::create(s);
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
            //println!("{}, {}", i, l);
        }
        fo.order = fo.order[ii..].to_vec();
        fo
    }
}

/**
 * トライ木
 */
struct Trie {
    alphabet: Vec<char>,
    trans: HashMap<(usize, char), usize>,
    position: Vec<Vec<(usize, usize)>>,
    txt: Vec<Vec<char>>,
}

impl Trie {
    /**
     * 文字列からトライ木を構築
     */
    pub fn new(s: Vec<Vec<char>>) -> Trie {

    //    s.push(Bytes::from(&b"abcba"[..]));
    //    s.push(Bytes::from(&b"abbac"[..]));
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
            position: dist,
            txt: s
        }
    }
}

