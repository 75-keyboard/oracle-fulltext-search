use std::collections::{ BTreeMap, HashMap, HashSet, BinaryHeap };
use std::cmp::{ Ordering, max, min };
use std::fs::read_to_string;
use std::fs::File;
use std::io::{self, BufRead, BufReader};
use bit_vec::BitVec;

use std::time::{Instant, Duration};

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

    //for i in 0..100 {
    //    s.push(read_to_string(format!("data/txt/man{}.txt", i)).unwrap().chars().collect());
    //}
    //s.push(read_to_string("data/manual.txt").unwrap().chars().collect::<Vec<char>>());
    // for result in BufReader::new(File::open("data/test.txt").unwrap()).lines() {
    //     s.push(result.unwrap().chars().collect());
    // }
//    println!("{:?}", s);
    //for i in 0..517431 {

    //// 単語数に対する構築時間実験
    //println!("ファイル数, 単語数, 文字数, 構築時間, 検索時間(10回平均)");
    //println!("ファイル数, 状態数, 統合数");
    //let mut i = 1;
    //'outer: loop {
    //    for j in 0..9 {
    //        if (j+1)*i > 517431 { break 'outer; }
    //        exec1((j+1)*i);
    //    }
    //    i *= 10;
    //}

    // パターン長に対する検索時間実験
    exec2(100000);
    //exec2(531);
    //create_index(fa);
    //exec3();
}

fn exec3() {
    let oracle = FullTextFactorOracle::new({
        let mut s: Vec<Vec<char>> = Vec::new();
        let mut sum = 0;
        for i in 0..100 {
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
        println!("moji {}", sum);
        s
    });
}

fn exec2(idx: usize) {
    let mut max = 0;
    let mut min = 1_000_000_000;
    let mut num_words = 0;
    let mut num = 0;
    // ファクターオラクルの遷移のラベルの出現頻度からハフマン符号を構成
    let start = Instant::now();
    let mut ss: HashMap<usize, Vec<usize>> = HashMap::new();
    let oracle = FullTextFactorOracle::new({
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
    //let end = start.elapsed();
    print!("{}, {}, {}, ", idx, num_words, num);

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
            let s = oracle.txt[ss[&i][j]].clone()[0..x].iter().collect::<String>();
            //println!("{}, {}", x, s);

            let start = Instant::now();
            let result = oracle.search(s);
            let end = start.elapsed();
            sum += end.as_micros();
        }
        println!("{}, {}", x, sum/1000);
    }
}

fn exec1(idx: usize) {
    let mut num_words = 0;
    let mut num = 0;
    // ファクターオラクルの遷移のラベルの出現頻度からハフマン符号を構成
    let start = Instant::now();
    print!("{}, ", idx);
    let oracle = Trie::new({
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
    //print!("{}, {}, {}, {}, ", idx, num_words, num, end.as_micros());
    ////let fa = FullTextFactorOracle::new_both({
    ////    let mut s: Vec<Vec<char>> = Vec::new();
    ////    //for i in 0..1000 {
    ////    //    s.push(read_to_string(format!("data/txt/man{}.txt", i)).unwrap().chars().collect());
    ////    //}
    ////    //s.push("abbac".chars().collect());
    ////    //s.push("aaaac".chars().collect());
    ////    //
    ////    println!("Files loaded.");

    ////    s
    ////});
    ////println!("{:?}", fa);

    ////println!("{:?}", fa.search("abbac".to_string()));
    ////println!("{:?}", fa.search("html".to_string()).len());
    ////println!("{:?}", fa.search("test".to_string()));

    //use rand::Rng;
    //let mut sum = 0;
    //for _ in 0..1000 {
    //    //let s = {
    //    //    let mut s = String::new(); // バッファを確保
    //    //    std::io::stdin().read_line(&mut s).unwrap(); // 一行読む。失敗を無視
    //    //    s.trim_end().to_owned() // 改行コードが末尾にくっついてくるので削る
    //    //};
    //    use rand::Rng;
    //    let mut rng = rand::thread_rng(); // デフォルトの乱数生成器を初期化します
    //    let i: usize = rng.gen();           // genはRng traitに定義されている
    //    let i: usize = i % num_words;           // genはRng traitに定義されている
    //    let s = oracle.txt[i as usize].clone();

    //    let start = Instant::now();
    //    let result = oracle.search(s.iter().collect::<String>());
    //    let end = start.elapsed();
    //    sum += end.as_micros();
    //    //println!("{:?}", result);
    //    //println!("{:?} {}.{:03}", result.len(), end.as_secs(), end.subsec_nanos() / 1_000_000);
    //}
    //println!("{}", sum/1000);
}

fn create_index(fa: FullTextFactorOracle) {
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
#[derive(Debug)]
struct Trie {
    alphabet: Vec<char>,
    //trans: Vec<HashMap<char, usize>>,
    trans: HashMap<(usize, char), usize>,
    position: Vec<Vec<(usize, usize)>>,
    txt: Vec<Vec<char>>,
}

impl Trie {
    /**
     * 文字列からトライ木を構築
     */
    pub fn new(s: Vec<Vec<char>>) -> Trie {
        let mut trie_trans = HashMap::new();
        let mut reverse_trans: BTreeMap<usize, (char, usize)> = BTreeMap::new();
        let mut trie_alphabet = HashSet::new();
        let mut finals = HashSet::new();
        let mut last = 0;
        let mut dist: Vec<Vec<(usize, usize)>> = Vec::new();
        dist.push(Vec::new());
        let mut summ = 0;
        for i in 0..s.len() {
            dist[0].push((i, 0));
            let mut crt = 0;
            for j in 0..s[i].len() {
                trie_alphabet.insert(s[i][j]);
                match trie_trans.get(&(crt, s[i][j])) {
                    Some(x) => {
                        crt=*x;
                        dist[crt].push((i, j+1));
                    },
                    None => {
                        summ += 1;
                        dist.push(Vec::new());
                        dist[last+1].push((i, j+1));
                        last+=1;
                        trie_trans.insert((crt, s[i][j]), last);
                        reverse_trans.insert(last, (s[i][j], crt));
                        crt=last;
                        if j == s[i].len()-1 { finals.insert(crt); } else { finals.remove(&crt); }
                    }
                }

                //match trans[ccrt].get(&s[i][j]) {
                //    Some(x) => {
                //        ccrt=*x;
                //        dist[ccrt].push((i, j+1));
                //    },
                //    None => {
                //        llast+=1;
                //        dist.push(Vec::new());
                //        dist[llast].push((i, j+1));
                //        trans[ccrt].insert(s[i][j], llast);
                //        trans.push(HashMap::new());
                //        ccrt=llast;
                //    }
                //}
            }
        }
        for trans in trie_trans.iter() {
            finals.remove(&(trans.0).0);
        }

        println!("SUMMARY {}", summ);

        let mut pair_sum = finals.len() - 1;
        let mut next = vec![finals.into_iter().collect::<Vec<usize>>()];
        println!("num {}", next[0].len());
        let mut mm: HashMap<usize, Vec<usize>> = HashMap::new();
        mm.insert(*next[0].iter().min().unwrap(), next[0].clone());

        'a: loop {
            let mut tmp: Vec<Vec<usize>> = Vec::new();

            //println!("{:?}", next);
            for j in 0..next.len() {
                let mut hm: HashMap<char, Vec<usize>> = HashMap::new();
                for i in 0..next[j].len() {
                    //println!("aa {:?}", next);
                    match reverse_trans.get(&next[j][i]) {
                        Some(x) => { 
                            match hm.get_mut(&x.0) {
                                Some(y) => {
                                    y.push(x.1);
                                },
                                None => {
                                    hm.insert(x.0, vec![x.1]);
                                }
                            }          
                        }, None => {},
                    };
                }

                hm.retain(|_, v| v.len() > 1);
                for k in hm {
                    pair_sum += k.1.len() - 1;
                    mm.insert(*k.1.iter().min().unwrap(), k.1.clone());
                    tmp.push(k.1);
                }
            }
            next = tmp;
            if next.len() == 0 { break 'a; }
//            next = Vec::new();
//            hm.retain(|&_, v| v.len() != 1);
//            if hm.len() == 0 { break 'a; } else {
//                for k in hm {
//                    pair_sum += k.1.len() - 1;
//                    mm.insert(*k.1.iter().min().unwrap(), k.1.clone());
//                    next.push(k.1);
//                }
//            }
        }
        println!("pair {:?}", pair_sum);
        println!("{:?}", mm.len());
        let mut a :HashSet<usize> = HashSet::new();
        let mut b :Vec<usize> = Vec::new();
        for i in mm {
            for j in i.1 {
                b.push(j.clone());
                a.insert(j);
            }
        }
        println!("{} {}", a.len(), b.len());

        let mut trie_alphabet = trie_alphabet.into_iter().collect::<Vec<char>>();
        trie_alphabet.sort();
        //println!("Trie constructed.");

        Trie {
            alphabet: trie_alphabet,
            trans: trie_trans,
            //trans: trans,
            position: dist,
            txt: s,
        }
    }
}


#[derive(Debug)]
struct FullTextFactorOracle {
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

impl FullTextFactorOracle {
    /**
     * 初期化
     */
    fn init_both() -> FullTextFactorOracle {
        let supply_function :Vec<usize> = vec![1_000_000_000_000];
        let trans: Vec<HashMap<char, usize>>= vec![HashMap::new()];
        FullTextFactorOracle { 
            state_num: 0,
            trans: trans,
            supply_function,
            suffix_link_tree: vec![Vec::new()],
            searching_info: (Vec::new(), Vec::new()),
            sp_inverse: Vec::new(),
            position: Vec::new(),
            occurence: HashMap::new(),
            txt: Vec::new(),
        }
    }
    /**
     * Allauzenのオンラインファクターオラクル構成法
     */
    pub fn new_both(s: Vec<Vec<char>>) -> FullTextFactorOracle {
        //let p :Vec<char> = "abbbaab".chars().collect();

        let mut oracle = FullTextFactorOracle::init_both();
        let mut trie_trans = HashMap::new();
        let mut trie_alphabet = HashSet::new();
        let mut last = 0;
        let mut dist: Vec<Vec<(usize, usize)>> = Vec::new();
        dist.push(Vec::new());
        let mut summ = 0;
        for i in 0..s.len() {
            dist[0].push((i, 0));
            let mut crt = 0;
            summ += s[i].len();
            for j in 0..s[i].len() {
                trie_alphabet.insert(s[i][j]);
                let x = match trie_trans.get(&(crt, s[i][j])) {
                    Some(x) => {
                        dist[*x as usize].push((i, j+1));
                        //if j == s[i].len()-1 { finals.push(crt-1) }
                        *x
                    },
                    None => {
                        dist.push(Vec::new());
                        dist[last+1].push((i, j+1));
                        last+=1;
                        trie_trans.insert((crt, s[i][j]), last);
                        //reverse_trans.insert(last, (s[i][j], crt));
                        //if j == s[i].len()-1 { finals.push(crt) }

                        oracle.trans.push(HashMap::new());
                        oracle.supply_function.push(0);
                        oracle.suffix_link_tree.push(Vec::new());
                        last
                    }
                };

                oracle.add_state(crt, s[i][j], x);
                crt = x;
            }
        }
        //println!("SUMMARY {}", summ);


        let mut trie_alphabet = trie_alphabet.into_iter().collect::<Vec<char>>();
        trie_alphabet.sort();
        //println!("Trie constructed.");

        let trie = Trie {
            alphabet: trie_alphabet,
            trans: trie_trans,
            //trans: trans,
            position: dist,
            txt: s,
        };

        oracle.txt = trie.txt;
        oracle.position = trie.position;
        oracle.organize_seaching_info();

        //println!("Oracle constructed");
        //println!("{:?}", oracle);
        oracle
    }

    /**
     * Allauzenのオンラインファクターオラクル構成法
     */
    pub fn new(s: Vec<Vec<char>>) -> FullTextFactorOracle {
        //let p :Vec<char> = "abbbaab".chars().collect();
        let trie = Trie::new(s);

        let mut oracle = FullTextFactorOracle::init(trie.trans.len()+1, trie.position.clone(), trie.txt);
        //oracle.depth(0, &trie.trans);

        //for i in 0..oracle.txt.len() {
        //    let mut src = 0;
        //    for j in 0..oracle.txt[i].len() {
        //        src = oracle.add_state(src, oracle.txt[i][j], oracle.state_num);
        //    }
        //}

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

        //println!("Oracle constructed");
        //println!("{:?}", oracle);
        oracle
    }

    fn depth(&mut self, num: usize, trie: &Vec<HashMap<char, usize>>) {
        for i in &trie[num] {
            self.add_state(num, *i.0, *i.1);
            self.depth(*i.1, trie);
        }
    }

    fn organize_seaching_info(&mut self) {
        let searching_info = self.re_order(0, Vec::new(), vec![0; self.supply_function.len()], vec![(0, 0); self.supply_function.len()]);


        self.searching_info = (searching_info.0, searching_info.1);
        self.sp_inverse = searching_info.2;
    }

    /**
     * 初期化
     */
    fn init(len: usize, position: Vec<Vec<(usize, usize)>>, txt: Vec<Vec<char>>) -> FullTextFactorOracle {
        let mut supply_function :Vec<usize> = vec![0; len];
        supply_function[0] = 1_000_000_000_000;      
        let trans: Vec<HashMap<char, usize>>= vec![HashMap::new(); len];
        FullTextFactorOracle { 
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
    fn add_state(&mut self, src: usize, sigma: char, dst: usize) -> usize {
        if let Some(x) = self.trans[src].get(&sigma) {
            return *x;
        }

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

        return dst;
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

        // FOが受理したか
        if current_state == 1_000_000_000_000 {
            return endpos;
        }
        //println!("Accepted {:?} by {} is {}", w, current_state, self.searching_info.1[current_state]);

        // 受理した状態に対応する位置情報を確認
        for position in self.position[current_state].clone() {
            //println!("{:?}", position);
            let ww: Vec<char> = w.chars().collect();
            if ww.len() > position.1 { continue; }
            for i in 0 .. ww.len() {
                //println!("{} + {} - {}", position.1, i ,ww.len());
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
        //println!("S_INVERSE {:?}", s_inverse);

        let ww: Vec<char> = w.chars().collect();
        for j in s_inverse {
            //println!("{:?}",self.position[self.searching_info.0[j]]);
            for position in self.position[self.searching_info.0[j]].clone() {
                if ww.len() > position.1 { continue; }
                //println!("{:?}", position);
                for i in 0 .. ww.len() {
                    //println!("{} + {} - {}", position.1, i ,ww.len());
                    if ww[i] != self.txt[position.0][position.1 + i - ww.len()] {
                        break;
                    } else if i == ww.len() - 1 { 
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
                break;
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
        let fa = FullTextFactorOracle::new(vec!["abbac".chars().collect(), "aaaac".chars().collect()]);

        //println!("{:?}", fa.search("abbac".to_string()));
        assert_eq!(vec![(1, 2), (1, 3), (1, 4)], fa.search("aa".to_string()));
        assert_eq!(vec![(0, 2), (0, 3)], fa.search("b".to_string()));
        assert_eq!(vec![(0, 5), (1, 5)], fa.search("ac".to_string()));
    }
}
