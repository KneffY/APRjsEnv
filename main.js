// MODULES <<===================================

// VARIABLES <<===================================

let Env = {
    dom: document.getElementById ("env"),
    floor: 1070,
    screenCenter: [
        400, // Env.dom.innerWidth / 2, // from upper left corner
        250 // Env.dom.innerHeight / 2
    ],
    G: 3, // slower than every movement & still perceptible
    t: 0, // time exe
    blocks: [],
    R: 0, // por iteracion\\\nope  0.01/5ms
    epi: 0, // done or not done
    mr: 0, // wtf
    nexEpisode: () => {
        // Env.epi++;
        // R = 0; // reset reward
        Env.mr = 1;


        window.location.reload();
    }
}

let UI = {
    bb1: document.getElementById ("bb1"),
    bb2: document.getElementById ("bb2"),
    bb3: document.getElementById ("bb3"),
    ww1: document.getElementById ("ww1"),
    rwrd: document.getElementById ("rwrd"),
    episode: document.getElementById ("episode"),
}

let Block = class {
    constructor () {
        this.size = 100;
        this.XY = [0,0];
        this.DOM = "";
    }
    create (x,y) {
        let B = document.createElement('div'); B.classList.add ("block");
        Env.dom.appendChild(B); this.DOM = B;
        this.setPosition (x,y);
    }
    setPosition (x, y) {
        this.DOM.style.left = (x - this.size/2) + "px";
        this.DOM.style.top = (y - this.size/2) + "px";
        this.XY = [x - this.size/2, y - this.size/2]
    }
}

let Gol = {
    DOM:"",
    XY: [0,0],
    set: (x,y) => {
        let X = document.createElement('div'); X.classList.add ("gol");
        Gol.DOM = X;
        Env.dom.appendChild(X);
        Gol.DOM.style.left = (x - 75) + "px";
        Gol.DOM.style.top = (y - 75) + "px";
        Gol.XY = [x,y];
    },
}



let Agent = {
    dom: document.getElementById ("agent"), // core tag
    turbines: [ // dom objects, left and right turbines

        {
            DOM: document.getElementById ("LT"),
            ang: 0,
            rot: (s) => {
                Agent.turbines[0].ang += Agent.rotSpeed * s; // speed/side + v -
                Agent.turbines[0].DOM.style.transform = `rotate(${Agent.turbines[0].ang}deg)`;

                Agent.fV =[
                    A2V(Agent.turbines[0].ang-90, 1)[0]+ A2V (Agent.turbines[1].ang-90, 1)[0],
                    A2V(Agent.turbines[0].ang-90, 1)[1]+ A2V (Agent.turbines[1].ang-90, 1)[1],
                ]

                console.log ("left turbine", A2V(Agent.turbines[0].ang-90,1));
                
            },
        }
        ,
        {
            DOM: document.getElementById ("RT"),
            ang: 0,
            rot: (s) => {
                Agent.turbines[1].ang += Agent.rotSpeed * s; // speed/side + v -
                Agent.turbines[1].DOM.style.transform = `rotate(${Agent.turbines[1].ang}deg)`;
                Agent.fV =[
                    A2V(Agent.turbines[0].ang-90, 1)[0]+ A2V (Agent.turbines[1].ang-90, 1)[0],
                    A2V(Agent.turbines[0].ang-90, 1)[1]+ A2V (Agent.turbines[1].ang-90, 1)[1],
                ]
                console.log ("right turbine", Agent.fV);
            },
        }

    ], 
    legs: [ // dom objects, left and right legs

        { // left leg
            inDOM: document.getElementById ("INL"),
            inAng: 0,
            outDOM: document.getElementById ("OUTL"),
            outAng: 0,
            base: document.getElementById ("Lbase"),
            baseH: 0, // distance from base to ground
            x: 0,
            inRot: (s) => {
                // dom.style.transform = `rotate(${ang}deg)`;
                Agent.legs[0].inAng += Agent.rotSpeed * s; // speed/side + v -
                Agent.legs[0].inDOM.style.transform = `rotate(${Agent.legs[0].inAng}deg)`;
                console.log ("inner left leg");
            },
            outRot: (s) => {
                Agent.legs[0].outAng += Agent.rotSpeed * s; // speed/side + v -
                Agent.legs[0].outDOM.style.transform = `rotate(${Agent.legs[0].outAng}deg)`;
                console.log ("outher left leg");
            },
        }
        ,
        { // right leg
            inDOM:document.getElementById ("INR"),
            inAng: 0,
            outDOM: document.getElementById ("OUTR"),
            outAng: 0,
            base: document.getElementById ("Rbase"),
            baseH: 0,
            x: 0,
            inRot: (s) => {
                // dom.style.transform = `rotate(${ang}deg)`;
                Agent.legs[1].inAng += Agent.rotSpeed * s; // speed/side + v -
                Agent.legs[1].inDOM.style.transform = `rotate(${Agent.legs[1].inAng}deg)`;
                console.log ("inner right leg");
            },
            outRot: (s) => {
                Agent.legs[1].outAng += Agent.rotSpeed * s; // speed/side + v -
                Agent.legs[1].outDOM.style.transform = `rotate(${Agent.legs[1].outAng}deg)`;
                console.log ("outher right leg");
            },
        }

    ], 
    legsPos: () => {
        // window.scrollY + document.querySelector('#elementId').getBoundingClientRect().top // Y
        Agent.legs.forEach ( (L)=> {
            L.baseH = Env.floor - L.base.getBoundingClientRect().top - 18;
            // console.log (Env.floor)
        })

        // window.scrollX + document.querySelector('#elementId').getBoundingClientRect().left // X
    },

    

    // var data
    battery: 100,
    speed: 4,
    rotSpeed: 3,
    flying: 0, // to mult turSpeed as module vec
    turSpeed: 2, // mod vec not unitary
    fV: [0,1],
    size: 100,
    angle: 0, // relative to 3 o clock
    height: 0, // relative to ground
    x: 0, // relative to screen
    y: 0, // relative to screen
    x_1: 0, // last position
    y_1: 0,
    ABL: 0,// angle between legs

    // auxiliar <<===========================
    WM: [0, 0, 0, 0], // Lin, Lout, Rin, Rout
    TM: [0,0], // LT RT spinning



    create: () => {
        // initialize html tags
        // Agent.dom.style.left = `${Env.screenCenter[0] - Agent.size/2}px`;
        // Agent.dom.style.top = `${Env.screenCenter[1] - Agent.size/2}px`;
        Agent.x = Env.screenCenter[0] - Agent.size/2 - 150;
        Agent.y = Env.screenCenter[1] - Agent.size/2;
        console.log (Env.screenCenter[0])
    },
    collide: () => {
        // agent collides not legs nor turbines
    },
    rotate: (s) => {
        Agent.angle += Agent.rotSpeed * s; // speed/side + v -
        Agent.dom.style.transform = `rotate(${Agent.angle}deg)`;
    },
    move: (v, s) => {
        // Agent.dom.style.left = `${Agent.x + v[0]* Agent.speed - Agent.size/2}px`;
        // Agent.dom.style.top = `${Agent.y - v[1]* Agent.speed - Agent.size/2}px`;


        // let A = Agent.angle - V2A(v);
        // console.log (V2A(v), Agent.angle);
        // v = A2V (A, 1);
        Agent.x_1 = Agent.x;
        Agent.y_1 = Agent.y;
        Agent.dom.style.left = `${Agent.x + v[0]* s}px`;
        Agent.dom.style.top = `${Agent.y - v[1]* s}px`;
        Agent.x = Agent.x + v[0] * s;
        Agent.y = Agent.y - v[1] * s;

        // console.log (Agent.x + v[0]* Agent.speed,Agent.y - v[1]* Agent.speed);
        // console.log (Agent.x);
    },
}




// FUNCTIONS <<===================================

let A2V = (a,m) => { // unitary
    let x = m * Math.cos ((a * Math.PI)/180);
    let y = (-1) * m * Math.sin ((a * Math.PI)/180);
    let M = Math.sqrt (x*x + y*y);
    return [x/M,y/M];
}

let V2A = (v) => { 
    return (Math.atan (v[1]/v[0]) * 180) / Math.PI;
}

let WIN = () => {
    let x1 = Agent.x;
    let y1 = Agent.y;
    let x2 = Gol.XY[0];
    let y2 = Gol.XY[1]-100;
    if (Math.sqrt ((x1-x2)**2 + (y1-y2)**2) < 250 && Env.mr == 0) {

        // winning thing 
        // alert("NICE");
        console.log ("WON");
        Env.mr = 1;
        Env.R+= 10;
    }
}



// EVENTS <<===================================
document.addEventListener("keydown", (e) => {
    // testing legs

    // LEFT LEG #######################################################
    if (e.code == "KeyA") { // rotate positive left inner leg
        Agent.WM[0] = 1;
    } 
    if (e.code == "KeyZ") { // rotate negative left inner leg
        Agent.WM[0] = -1;
    } 
 
    //  outher left leg
    if (e.code == "KeyS") { // rotate positive left outher leg
        Agent.WM[1] = 1;
    } 
    if (e.code == "KeyX") { // rotate negative left outher leg
        Agent.WM[1] = -1;
    } 

    // RIGHT LEG #######################################################
    if (e.code == "KeyD") { // rotate positive right inner leg
        Agent.WM[2] = 1;
    } 
    if (e.code == "KeyC") { // rotate negative right inner leg
        Agent.WM[2] = -1;
    } 
 
     //  outher left leg
    if (e.code == "KeyF") { // rotate positive right outher leg
        Agent.WM[3] = 1;
    } 
    if (e.code == "KeyV") { // rotate negative right outher leg
        Agent.WM[3] = -1;
    } 

    // CORE #######################################################
    if (e.code == "KeyG") { // rotate positive core
        Agent.rotate(1);
    } 
    if (e.code == "KeyB") { // rotate negative core
        Agent.rotate(-1);
    }

    // TURBINES #######################################################
    if (e.code == "KeyH") { // rotate positive left turbine
        Agent.TM[0]=1;
    } 
    if (e.code == "KeyN") { // rotate negative left turbine
        Agent.TM[0]=-1;
    }
    // right turbine
    if (e.code == "KeyJ") { // rotate positive left turbine
        Agent.TM[1]=1;
    } 
    if (e.code == "KeyM") { // rotate negative left turbine
        Agent.TM[1]=-1;
    }

    // START #######################################################
    if (e.code == "Space") { // start this... thing
        // comentar dsps
        Agent.create();
        Env.blocks.push (new Block());
        Env.blocks.push (new Block());
        Env.blocks.push (new Block());
        Env.blocks.forEach((e) => {e.create(
            Math.random() * 800 +450,
            Math.random() * 800 +100
            )
        });
        Gol.set (
            Math.random() * 600 +700,
            Math.random() * 400 +100
        )
    }

    // FLY WHERE SUM TURB AIMS #######################################################
    if (e.code == "ArrowUp") { // hopefully.. fly :,c
        // Agent.move(Agent.fV, Agent.speed);
        Agent.flying = 1;
        // console.log (Agent.fV);
    }
    
}) 
document.addEventListener("keyup", (e) => {
    // articulaciones
    if (e.code == "ArrowUp") {
        Agent.flying = 0;
    }
    if (e.code == "KeyA" || e.code == "KeyZ") {
        Agent.WM[0] = 0;
    }
    if (e.code == "KeyS" || e.code == "KeyX") {
        Agent.WM[1] = 0;
    }
    if (e.code == "KeyD" || e.code == "KeyC") {
        Agent.WM[2] = 0;
    }
    if (e.code == "KeyF" || e.code == "KeyV") {
        Agent.WM[3] = 0;
    }
    // turbinas
    if (e.code == "KeyH" || e.code == "KeyN") { // rotate positive left turbine
        Agent.TM[0] = 0;
    } 
    // right turbine
    if (e.code == "KeyJ" || e.code == "KeyM") { // rotate positive left turbine
        Agent.TM[1] = 0;
    } 
    
    // refresh
    if (e.code == "KeyR") { // rotate positive left turbine
        Env.nexEpisode();
    } 
})




// RUNNING <<===================================
console.log ("start :'D")
// alert ("wena");

Agent.create();
Env.blocks.push (new Block());
Env.blocks.push (new Block());
Env.blocks.push (new Block());
Env.blocks.forEach((e) => {e.create(
    Math.random() * 900 +550,
    Math.random() * 800 +100
    )
});
Gol.set (
    Math.random() * 600 +700,
    Math.random() * 400 +100
);

setInterval(() => {
    // time control
    Env.t++;

    if (Env.t == 400) {Env.mr = 1;} // stop add nor quiting reward

    WIN();
    // (Env.t > 200*3 && Env.mr == 0) ? Env.nexEpisode(): null;
    
    // screen center refresh
    // Env.screenCenter = [
    //         Env.dom.innerWidth / 2, // from upper left corner
    //         Env.dom.innerHeight / 2
    //     ]

    // floor refresh
    // Env.floor = Env.screenCenter[1]+190;
    // Env.floor = 350+190;

    // reward
    if (Math.sqrt ((Agent.x-Gol.XY[0])**2 + (Agent.y-Gol.XY[1])**2) > Math.sqrt ((Agent.x_1-Gol.XY[0])**2 + (Agent.y_1-Gol.XY[1])**2)) {
        (Env.mr == 0) ? Env.R -= 0.01: null;
    } else if (Math.sqrt ((Agent.x-Gol.XY[0])**2 + (Agent.y-Gol.XY[1])**2) < Math.sqrt ((Agent.x_1-Gol.XY[0])**2 + (Agent.y_1-Gol.XY[1])**2)) {
        (Env.mr == 0) ? Env.R += 0.01: null;
    }
    

    // AUXILIO
    

    UI.bb1.innerText = `${Math.sqrt ((Agent.x-Env.blocks[0].XY[0])**2 + (Agent.y-Env.blocks[0].XY[1])**2)}`;
    UI.bb2.innerText = `${Math.sqrt ((Agent.x-Env.blocks[1].XY[0])**2 + (Agent.y-Env.blocks[1].XY[1])**2)}`;
    UI.bb3.innerText = `${Math.sqrt ((Agent.x-Env.blocks[2].XY[0])**2 + (Agent.y-Env.blocks[2].XY[1])**2)}`;
    UI.ww1.innerText = `${Math.sqrt ((Agent.x-Gol.XY[0])**2 + (Agent.y-Gol.XY[1])**2)}`;
    UI.rwrd.innerText = `${Env.R}`;
    UI.episode.innerText = `${Env.mr}`;

    // TGO Q MEJORAR... EL CONTROL DEL MOVIMIENTO D:
    // L E G S :D
    if (Agent.WM[0] == 1) {
        Agent.legs[0].inRot(1);
    }
    if (Agent.WM[0] == -1) {
        Agent.legs[0].inRot(-1);
    }
    if (Agent.WM[1] == 1) {
        Agent.legs[0].outRot(1);
    }
    if (Agent.WM[1] == -1) {
        Agent.legs[0].outRot(-1);
    }
    if (Agent.WM[2] == 1) {
        Agent.legs[1].inRot(1);
    }
    if (Agent.WM[2] == -1) {
        Agent.legs[1].inRot(-1);
    }
    if (Agent.WM[3] == 1) {
        Agent.legs[1].outRot(1);
    }
    if (Agent.WM[3] == -1) {
        Agent.legs[1].outRot(-1);
    }
    // V O L A R
    if (Agent.flying == 1) {
        Agent.move(Agent.fV, Agent.speed);
    } 
    // C A E R
    if (Agent.legs[0].baseH > 4 && Agent.legs[0].baseH > 4 && Agent.flying == 0) {
        Agent.move ([0,-1], Env.G);
    }
    // T U R B I N A S spin
    if (Agent.TM[0] == 1) {
        Agent.turbines[0].rot(1);
    }
    if (Agent.TM[0] == -1) {
        Agent.turbines[0].rot(-1);
    }
    if (Agent.TM[1] == 1) {
        Agent.turbines[1].rot(1);
    }
    if (Agent.TM[1] == -1) {
        Agent.turbines[1].rot(-1);
    }


    // colision
    Env.blocks.forEach ((e) => {
        // let x1 = Agent.dom.getBoundingClientRect().left;
        // let y1 = Agent.dom.getBoundingClientRect().top;
        // let x2 = e.DOM.getBoundingClientRect().left;
        // let y2 = e.DOM.getBoundingClientRect().top;
        let x1 = Agent.x;
        let y1 = Agent.y;
        let x2 = e.XY[0];
        let y2 = e.XY[1];
        if (Math.sqrt ((x1-x2)**2 + (y1-y2)**2) < 150 && Env.mr == 0) {
            Env.mr = 1;
            Env.R = -5;
        }
    })

    // win ??

    

    // agent falling
    Agent.legsPos() // getting height from feets
    if (Env.t % 100 == 0) {
        
        Agent.legs.forEach ( (L)=> {
            // console.log (L.baseH );
        })
    }

    

    // AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAYUDAAAAAAAAAAAAAAAA
    if (Agent.legs[0].baseH < 4 && Agent.legs[1].baseH > 4) {
        //  /----
        Agent.rotate (1);
        Agent.move([1,0],Agent.speed);
    }
    if (Agent.legs[0].baseH > 4 && Agent.legs[1].baseH < 4) {
        //  ----\
        Agent.rotate (-1);
        Agent.move([-1,0],Agent.speed);
    }



},10);