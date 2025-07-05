var canvas = document.getElementById("marker");
var ctx = canvas.getContext("2d");
var width = canvas.width;
var height = canvas.height;
var curX, curY, prevX, prevY;
var play = true;
var hold = false
var parking_spots = []
var current_parking_spot = []
var parked_cars = []
var image = document.getElementById("source");
image.onload = drawImage

$(document).ready(function () {
   selectCamera()
   setInterval(function(){
        getParkedCards()
        console.log(parked_cars.length)

    }, 1000);
});


function drawImage() {
    canvas.width = image.naturalWidth;
    canvas.height = image.naturalHeight;
    ctx.drawImage(image, 0, 0, image.width, image.height);
    width = canvas.width;
    height = canvas.height;
    ctx.lineWidth = 2;
    ctx.strokeStyle = '#00FF00'; 
    draw() 
    if ( play ) {
        requestAnimationFrame(drawImage);
    }
}

function selectCamera() {
    var selected_camera = $('#camera_selected');
    $.ajax({
            url: '/select_camera',
            type: 'POST',
            data: JSON.stringify(selected_camera.val()),
            contentType: 'application/json;charset=UTF-8',
            cache:false,
            error: function(response){
                //alert('Cannot select the camera')
            },
            success: function(response){
                var img_src = $("#source").attr("src");
                var d = new Date();
                $("#source").attr("src", img_src+"?"+d.getTime());
                image = $('#source')[0];
                parking_spots = response
                drawImage()
            }
        });
}

function getParkedCards() {
    $.ajax({
            url: '/get_parked_cars',
            type: 'POST',
            contentType: 'application/json;charset=UTF-8',
            cache:false,
            error: function(response){
               // alert('Cannot get the parked cards')
            },
            success: function(response){
                parked_cars = response
            }
        });
}

function draw() {
    if ( width != 0 && height != 0 ) {
        img = ctx.getImageData(0, 0, width, height);
        ctx.lineWidth = 2;
        var free_ps = 0
        parking_spots.forEach(function (parking_spot) {
            if ( isParkingSpotFree(parking_spot) ) {
                ctx.strokeStyle = '#00FF00';
                free_ps += 1
            }
            else {
                ctx.strokeStyle = '#FF0000';
            }
            for (var i = 0; i < parking_spot.length-1; i++) {
                ctx.beginPath();
                ctx.moveTo(parking_spot[i][0], parking_spot[i][1]);
                ctx.lineTo(parking_spot[i+1][0], parking_spot[i+1][1]);
                ctx.stroke();
                ctx.closePath();
            }

        });
        ctx.putImageData(img, width, height);
        document.querySelector('#counter').innerHTML = free_ps; 

    }
}

function addParkingSpot(ps) {
    for (var i=0; i < parking_spots.length; i++) {
        if ( parkingSpotsIntersect(ps, parking_spots[i]) ){
            drawImage()
            update()
            return 
        }
    }
    parking_spots.push(ps)
    drawImage()
    update()
}

function removeParkingSpot(ps) {

    parking_spots = parking_spots.filter(function (item) {
        return item != ps;
    });
    drawImage()
    update()
}

function getParkingSpot(x,y) {
    for (var i=0; i < parking_spots.length; i++) {
        var pt = turf.point([x, y]);
        var poly = turf.polygon([parking_spots[i]]);
        if (turf.booleanPointInPolygon(pt, poly)) {
            return parking_spots[i]
        }
    }
};

function isInsideParkingSpot(ps, x,y) {
    var pt = turf.point([x, y]);
    var poly = turf.polygon([ps]);
    if (turf.booleanPointInPolygon(pt, poly)) {
        return true
    }
    return false
};

function isParkingSpotFree(ps) {
    for (var i=0; i < parked_cars.length; i++) {
        pc = parked_cars[i];
        center_x = pc[0][0] + ( pc[1][0] - pc[0][0] )/2
        center_y = pc[0][1] + ( pc[3][1] - pc[0][1] )/2
        if ( isInsideParkingSpot(ps, center_x, center_y ) ) {
            return false
        }

    }
    return true
}

function parkingSpotsIntersect(ps1,ps2) {
    var poly1 = turf.polygon([ps1]);
    var poly2 = turf.polygon([ps2]);

    return turf.intersect(poly1, poly2) != null;
};

function update(reset=false) {
    if ( parking_spots.length > 0) {
        document.querySelector('#remove').disabled = false;
        document.querySelector('#reset').disabled = false;
    }
    else {
        document.querySelector('#remove').disabled = true;   
        document.querySelector('#reset').disabled = true;

    }
    document.querySelector('#save').disabled = reset; 
}


function add() {
    play = false
    canvas.onmousedown = function (e) {
        img = ctx.getImageData(0, 0, width, height);
        prevX = e.clientX - canvas.offsetLeft;
        prevY = e.clientY - canvas.offsetTop;
        if (!hold) {
        	startX = prevX
        	startY = prevY
            current_parking_spot = []
            current_parking_spot.push([startX,startY]);
        }
        else {
            current_parking_spot.push([curX, curY]);
        }
        hold = true;
    };
            
    canvas.onmousemove = function linemove(e) {
        if (hold){
            ctx.putImageData(img, 0, 0);
            curX = e.clientX - canvas.offsetLeft;
            curY = e.clientY - canvas.offsetTop;
            draw()
            ctx.beginPath();
            ctx.moveTo(prevX, prevY);
            ctx.lineTo(curX, curY);
            ctx.stroke();
            ctx.closePath();
        }
    };
            
    canvas.onmouseup = function (e) {
        if (hold) {
            if (current_parking_spot.length == 4) {
                ctx.beginPath();
                ctx.moveTo(curX, curY);
                ctx.lineTo(startX, startY);
                ctx.stroke();
                ctx.closePath();    
                hold = false;
                current_parking_spot.push([startX,startY]);
                addParkingSpot(current_parking_spot)
                current_parking_spot = []
            }
        }
    };
}

function remove() {
    play = false
    canvas.onmousedown = function (e) {
        img = ctx.getImageData(0, 0, width, height);
        var x = e.clientX - canvas.offsetLeft;
        var y = e.clientY - canvas.offsetTop;
        ps = getParkingSpot(x,y)
        removeParkingSpot(ps) 
    };
}

function reset() {
    play = true
    parking_spots = []
    drawImage()
    update()
}

function save(){
    play = true
    drawImage()
    $.ajax({
            url: '/save',
            type: 'POST',
            data: JSON.stringify(parking_spots),
            contentType: 'application/json;charset=UTF-8',
            cache:false,
            error: function(response){
                alert('Error saving data')
            },
            success: function(response){
                document.querySelector('#save').disabled = true; 
            }
        });
} 


