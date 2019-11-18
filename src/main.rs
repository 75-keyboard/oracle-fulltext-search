use std::collections::{ HashMap, HashSet, VecDeque };

struct State {
    childs: Vec<State>,
    set: HashSet<usize>,
    position: HashSet<(usize, usize)>
}

impl State {
    fn has(&mut self, set: &HashSet<usize>) {
        for child in self.childs.iter_mut() {
            println!("{:?} {:?} {:?}", child.set, set, child.set.is_superset(&set));
            if child.set.is_superset(&set) {
                child.has(&set);
                return;
            }
        }
        self.childs.push(State {
            childs: Vec::new(),
            set: set.clone()
        });
    }

    fn print(&self, i: usize) {
        println!("{}: {:?}", i, self.set);
        for child in self.childs.iter() {
            child.print(i+1);
        }
        
    }
}

fn main(){
    let mut s = Vec::new();
    s.push("abcba".chars().collect::<Vec<char>>());
    s.push("abbac".chars().collect::<Vec<char>>());
    println!("{:?}", s);

    let mut h = HashMap::new();
    let mut hs = HashSet::new();
    let mut last = 0;
    let mut dist: Vec<Vec<(usize, usize)>> = Vec::new();
    for i in 0..s.len() {
        let mut crt = 0;
        for j in 0..s[i].len() {
            hs.insert(&s[i][j]);
            match h.get(&(crt, s[i][j])) {
                Some(x) => {
                    crt=*x;
                    dist[crt-1].push((i, j));
                },
                None => {
                    dist.push(Vec::new());
                    dist[last].push((i, j));
                    last+=1;
                    h.insert((crt, s[i][j]), last);
                    crt=last;
                }
            }
        }
    }

    println!("{:?}", h);
    println!("{:?}", hs);
    println!("{:?}", dist);

    let mut fa_states: Vec<Vec<usize>> = Vec::new();
    fa_states.push((0..last).collect::<Vec<usize>>());
    let mut state_set_tree = State {
        set: (0..last).collect::<HashSet<usize>>(),
        childs: Vec::new()
    };
    println!("{:?}", state_set_tree.set);
    let mut fa_trans = HashMap::new();

    let mut i = 0;
    while fa_states.len() > i {
        'outer: for t in hs.iter() {
            let mut tmp = Vec::new();
            for s in fa_states[i].iter() {
                match h.get(&(*s, **t)) {
                    Some(x) => {
                        tmp.push(*x);
                    },
                    None => {}
                }
            }
            if tmp.len() > 0 {
                tmp.sort();
                'inner: for u in fa_states.iter() {
                    if tmp[0] == u[0] {
                        fa_trans.insert((fa_states[i].clone(), t), u.clone());
                        continue 'outer;
                    }
                }
                fa_trans.insert((fa_states[i].clone(), t), tmp.clone());
                fa_states.push(tmp.clone());
                state_set_tree.has(&tmp.into_iter().collect::<HashSet<usize>>());
            }
        }
        i+=1;
    }
    println!("{:?}", fa_states);
    println!("--------------------------------");
    println!("{:?}", fa_trans);

    state_set_tree.print(0);
}
